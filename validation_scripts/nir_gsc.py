import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import nir

# Citation: Network definition from neurobench Google Speech Commands example
spike_grad = surrogate.fast_sigmoid()
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameter
        beta = 0.9

        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(20, 256),
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        self.linear2 = nn.Linear(256, 256),
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        self.linear3 = nn.Linear(256, 256),
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        self.linear4 = nn.Linear(256, 35),
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)


def forward(self, x):
    # Layer 1
    y = self.flat1(x)
    y = self.linear1(y)
    spk1 = self.lif1(y)

    # Layer 2
    y = self.linear2(spk1)
    spk2 = self.lif2(y)

    # Layer 3
    y = self.linear3(spk2)
    spk3 = self.lif3(y)

    # Layer 4
    y = self.linear4(spk3)
    spk4, mem4 = self.lif4(y)   
    
    return spk4, mem4

if __name__ == "__main__":
    # Load the pre-trained weights
    net = Net()

    # First, load the sequential model state dict to inspect it
    sequential_state = torch.load(r"/home/ala4225/neurobench/neurobench/examples/gsc/model_data/s2s_gsc_snntorch",map_location=torch.device('cpu'))

    # Initialize NIR
    sample_data = torch.randn(1,1) # TODO: these dimensions are likely wrong
    nir_model = snn.export_to_nir(net, sample_data) # not supported RN?

    # Save to file
    nir.write("nir_model.nir", nir_model)

    print("Done!")