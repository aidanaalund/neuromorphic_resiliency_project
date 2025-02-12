import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

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
    l = list(net.lif1)
    print(l)


    # First, load the sequential model state dict to inspect it
    sequential_state = torch.load(r"/home/ala4225/neurobench/neurobench/examples/gsc/model_data/s2s_gsc_snntorch",map_location=torch.device('cpu'))
    # Print the keys to see the layer names
    for key in sequential_state.keys():
        print(key)
    # This might print something like:
    # 0.weight
    # 0.bias
    # 1.weight
    # 1.bias
    # etc.

    sequential_to_manual = {
        '1.weight' : 'flat1.weight',
        '1.bias' : 'flat1.bias',
        '2.threshold': 'lif1.threshold',
        '2.graded_spikes_factor' : 'lif1.graded_spikes_factor',
        '2.reset_mechanism_val' : 'lif1.reset_mechanism_val',
        '2.beta' : 'lif1.beta',
        '3.weight' : 'linear1.weight',
        '3.bias' : 'linear1.bias',
        '4.threshold' : 'lif2.threshold',
        '4.graded_spikes_factor' : 'lif2.graded_spikes_factor',
        '4.reset_mechanism_val' : 'lif2.reset_mechanism_val',
        '4.beta' : 'lif2.beta',
        '5.weight' : 'linear2.weight',
        '5.bias' : 'linear2.bias',
        '6.threshold' : 'lif3.threshold',
        '6.graded_spikes_factor' : 'lif3.graded_spikes_factor',
        '6.reset_mechanism_val' : 'lif3.reset_mechanism_val',
        '6.beta' : 'lif3.beta',
        '7.weight' : 'linear3.weight',
        '7.bias' : 'linear3.bias',
        '8.threshold' : 'lif4.threshold',
        '8.graded_spikes_factor' : 'lif4.graded_spikes_factor',
        '8.reset_mechanism_val' : 'lif4.reset_mechanism_val',
        '8.beta' : 'lif4.beta'
    }



    new_state_dict = {}
    for old_key, param in sequential_state.items():
        if old_key in sequential_to_manual:
            new_key = sequential_to_manual[old_key]
            new_state_dict[new_key] = param
    
    print(new_state_dict.keys())
    net.load_state_dict(new_state_dict)
    # TODO: figure out the input encoding

    # convert to yaml
    #utils.convert_class_snn_to_yaml(net,'gsc',[{'rate': 0.6}])