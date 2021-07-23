# PyTorch Implementation: RotNet
PyTorch implementation of [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
based on the [official implementation](https://github.com/gidariss/FeatureLearningRotNet).

The implementation supports the following datasets:
- CIFAR-10 / CIFAR-100
- SVHN
- Caltech101 / Caltech256
- STL10
- HAM10000
- ImageNet


## Installation
Required python packages are listed in `requirements.txt`. All dependencies can be installed using pip
```
pip install -r requirements.txt
```
or using conda
```
conda install --file requirements.txt
```

## Training
RotNet training is started by running the following command (`--pbar` to show progress bar during training):
```
python main.py --pbar
```
All commandline arguments, which can be used to adapt the configuration of the RotNet training are defined and described in `arguments.py`.
By default the following configuration is run:
```
model: 'NIN_4'
dataset: 'cifar10'
lr: 0.1
wd: 0.0005
epochs: 100
batch_size: 128
device: 'cuda'
out_dir: 'rotation_prediction'
momentum: 0.9
```
In addition to these, the following arguments can be used to further configure the RotNet training process:
* `--device <cuda / cpu>`: Specify whether training should be run on GPU (if available) or CPU
* `--num-workers <num_workers>`: Number of workers used by torch dataloader
* `--resume <path to run_folder>`: Resumes training of training run saved at specified path, e.g. `'out/rotnet_training/run_0'`. Dataset splits, model state, optimizer state, etc.
  are loaded and training is resumed with specified arguments.
* see `arguments.py` for more

Alternatively, the `polyaxon.yaml`-file can be used to start the RotNet training on a polyaxon-cluster:
```
polyaxon run -f polyaxon.yaml -u
```
For a general introduction to polyaxon and its commandline client, please refer to the [official documentation](https://github.com/polyaxon/polyaxon)
## Monitoring
The training progress (loss, accuracy, etc.) can be monitored using tensorboard as follows:
```
tensorboard --logdir <result_folder>
```
This starts a tensorboard instance at `localhost:6006`, which can be opened in any common browser.

## Evaluation
A trained RotNet model can be evaluated by running:
```
 python3 eval.py --run-path out/rotnet_training/run_0 --pbar --device <cuda / cpu>
```
where `--run-path` specifies the path at which the run to be evaluated is saved. Alternatively, one can also check all
metrics over all epochs using the tensorboard file.

## References
```
@inproceedings{
  gidaris2018unsupervised,
  title={Unsupervised Representation Learning by Predicting Image Rotations},
  author={Spyros Gidaris and Praveer Singh and Nikos Komodakis},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1v4N2l0-},
}
```