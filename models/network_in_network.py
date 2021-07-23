import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential()
        self.layers.add_module(
            "Conv",
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))
        self.layers.add_module("ReLU", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)


class NetworkInNetwork(nn.Module):
    def __init__(
            self, num_classes, num_stages=3, num_inchannels=3, use_avg_on_conv3=True
    ):
        super(NetworkInNetwork, self).__init__()

        assert num_stages >= 3
        n_channels = 192
        n_channels2 = 160
        n_channels3 = 96

        blocks = [nn.Sequential() for i in range(num_stages)]
        # 1st block
        blocks[0].add_module("Block1_ConvB1", BasicBlock(num_inchannels, n_channels, 5))
        blocks[0].add_module("Block1_ConvB2", BasicBlock(n_channels, n_channels2, 1))
        blocks[0].add_module("Block1_ConvB3", BasicBlock(n_channels2, n_channels3, 1))
        blocks[0].add_module(
            "Block1_MaxPool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2nd block
        blocks[1].add_module("Block2_ConvB1", BasicBlock(n_channels3, n_channels, 5))
        blocks[1].add_module("Block2_ConvB2", BasicBlock(n_channels, n_channels, 1))
        blocks[1].add_module("Block2_ConvB3", BasicBlock(n_channels, n_channels, 1))
        blocks[1].add_module(
            "Block2_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 3rd block
        blocks[2].add_module("Block3_ConvB1", BasicBlock(n_channels, n_channels, 3))
        blocks[2].add_module("Block3_ConvB2", BasicBlock(n_channels, n_channels, 1))
        blocks[2].add_module("Block3_ConvB3", BasicBlock(n_channels, n_channels, 1))

        if num_stages > 3 and use_avg_on_conv3:
            blocks[2].add_module(
                "Block3_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        for s in range(3, num_stages):
            blocks[s].add_module(
                "Block" + str(s + 1) + "_ConvB1", BasicBlock(n_channels, n_channels, 3)
            )
            blocks[s].add_module(
                "Block" + str(s + 1) + "_ConvB2", BasicBlock(n_channels, n_channels, 1)
            )
            blocks[s].add_module(
                "Block" + str(s + 1) + "_ConvB3", BasicBlock(n_channels, n_channels, 1)
            )

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module("GlobalAveragePooling", GlobalAveragePooling())
        blocks[-1].add_module("Classifier", nn.Linear(n_channels, num_classes))

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ["conv" + str(s + 1) for s in range(num_stages)] + [
            "classifier"
        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)

    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = (
            [self.all_feat_names[-1]] if out_feat_keys is None else out_feat_keys
        )

        if len(out_feat_keys) == 0:
            raise ValueError("Empty list of output feature keys.")
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    "Feature with name {0} does not exist. Existing features: {1}.".format(
                        key, self.all_feat_names
                    )
                )
            elif key in out_feat_keys[:f]:
                raise ValueError("Duplicate output feature key: {0}.".format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])
        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
              x: input image.
              out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        return out_feats
