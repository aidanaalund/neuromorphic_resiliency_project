import utils

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Citation: Network definition from neurobench Google Speech Commands example
beta = 0.9
spike_grad = surrogate.fast_sigmoid()
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(20, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 35),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
)

if __name__ == "__main__":
    # Load the pre-trained weights
    model_path = r"/home/ala4225/neurobench/neurobench/examples/gsc/model_data/s2s_gsc_snntorch"
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only = True))

    # convert to yaml
    utils.convert_sequential_snn_to_yaml(net,'gsc',[{'rate': 0.6}])