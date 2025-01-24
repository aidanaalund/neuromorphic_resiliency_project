import utils

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

hidden = 2
beta = 0.90
num_inputs = 2
num_outputs = 2

flat1 = nn.Flatten() # This will become NoneType in NIR
lin1 = nn.Linear(num_inputs,hidden)
lif1 = snn.Leaky(beta=beta, init_hidden=True) # num_neurons = hidden. Becomes nir.LIF
lin2 = nn.Linear(hidden,num_outputs)
lin2.weight = nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4]])) # the (i,j) entry is an edge from source i to target j
lif2 = snn.Leaky(beta=beta, init_hidden=True, output=True)

net = nn.Sequential(flat1,lin1,lif1,lin2,lif2)

if __name__ == "__main__":
    
    utils.convert_sequential_snn_to_yaml(net,'toy_example')
    
    # # Create sample input (batch_size=1, time_steps=1, input_size=num_inputs)
    # input_data = torch.ones((1, 1, num_inputs))
    
    # # Enable printing of intermediate values
    # for name, module in net.named_children():
    #     if isinstance(module, nn.Linear):
    #         print(f"\n{name} weights:")
    #         print(module.weight.detach())
    
    # # Forward pass
    # with torch.no_grad():
    #     output = net(input_data)
    #     print("\nOutput:", output)