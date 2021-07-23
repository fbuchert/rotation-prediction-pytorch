from functools import partial
from models.network_in_network import NetworkInNetwork


def get_network_in_network(num_classes, num_stages, **kwargs):
    return NetworkInNetwork(num_classes=num_classes, num_stages=num_stages, use_avg_on_conv3=False)


MODEL_GETTERS = {
    "NIN_3": partial(get_network_in_network, num_stages=3),
    "NIN_4": partial(get_network_in_network, num_stages=4),
    "NIN_5": partial(get_network_in_network, num_stages=5),
}
