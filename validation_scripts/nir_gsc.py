import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch.export_nir import export_to_nir
import nir

# Citation: Network definition from neurobench Google Speech Commands example
spike_grad = surrogate.fast_sigmoid()
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameter
        beta = 0.9


        # Initialize layers, but don't leave trailing comma for NIRTorch compatability
        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(20, 256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.linear2 = nn.Linear(256, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.linear3 = nn.Linear(256, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.linear4 = nn.Linear(256, 35)
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

    # # Load the pre-trained weights
    model_path = r"/home/ala4225/neurobench/neurobench/examples/gsc/model_data/s2s_gsc_snntorch"
    # net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only = True))# Load the sequential model state dict
    sequential_state = torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=True
    )

    # Create mapping from sequential to class model
    sequential_to_class = {
        '1.weight': 'linear1.weight',
        '1.bias': 'linear1.bias',
        '2.threshold': 'lif1.threshold',
        '2.graded_spikes_factor': 'lif1.graded_spikes_factor',
        '2.reset_mechanism_val': 'lif1.reset_mechanism_val',
        '2.beta': 'lif1.beta',
        '3.weight': 'linear2.weight',
        '3.bias': 'linear2.bias',
        '4.threshold': 'lif2.threshold',
        '4.graded_spikes_factor': 'lif2.graded_spikes_factor',
        '4.reset_mechanism_val': 'lif2.reset_mechanism_val',
        '4.beta': 'lif2.beta',
        '5.weight': 'linear3.weight',
        '5.bias': 'linear3.bias',
        '6.threshold': 'lif3.threshold',
        '6.graded_spikes_factor': 'lif3.graded_spikes_factor',
        '6.reset_mechanism_val': 'lif3.reset_mechanism_val',
        '6.beta': 'lif3.beta',
        '7.weight': 'linear4.weight',
        '7.bias': 'linear4.bias',
        '8.threshold': 'lif4.threshold',
        '8.graded_spikes_factor': 'lif4.graded_spikes_factor',
        '8.reset_mechanism_val': 'lif4.reset_mechanism_val',
        '8.beta': 'lif4.beta'
    }

    # Create new state dict with mapped keys
    new_state_dict = {}
    for old_key, param in sequential_state.items():
        if old_key in sequential_to_class:
            new_key = sequential_to_class[old_key]
            new_state_dict[new_key] = param

    # Load the mapped weights
    net.load_state_dict(new_state_dict)


    # Initialize NIR
    sample_data = torch.randn(1,1,20) # try (1,20) later
    nir_model = export_to_nir(net, sample_data)

    # TODO: fix save to file, not working at the moment...
    # nir.write("nir_model.nir", nir_model)

    # TODO: use this graph to validate your own yaml conversion (if helpful)

    print("Done!")