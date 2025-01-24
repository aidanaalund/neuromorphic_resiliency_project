import utils
import yaml

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Network definition from neurobench dvs_gesture example
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        beta_1 = 0.9999903192467171
        threshold_1 = 3.511291184386264
        spike_grad = surrogate.atan()
        
        # Initialize layers
        self.conv1 = nn.Conv2d(2, 16, 5, padding="same")
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta_1, threshold=threshold_1, spike_grad=spike_grad, init_hidden=True, output=True)

        # weights for conv2d layer, dimensions are (out, in/groups, kernel_size[0], kernel_size[1])
        with torch.no_grad(): # direct access to weight tensor
            self.conv1.weight[0, 0, 0, 0] = 0.1  # N, C, H, W
            self.conv1.weight[0, 1, 0, 0] = 0.2  # N, C, H, W
            self.conv1.weight[0, 0, 0, 1] = 0.3  # N, C, H, W
            self.conv1.weight[0, 1, 0, 1] = 0.4  # N, C, H, W

    def forward(self, x):
        # x is expected to be in shape (batch, channels, height, width) = (B, 2, 32, 32)
        
        # Layer 1
        y = self.conv1(x)
        y = self.pool1(y)
        spk1, mem1 = self.lif1(y)

        return spk1, mem1
    
if __name__ == "__main__":
    # initialize network
    net = Net()

    # create yaml file (must handle nn conv2d and nn maxpool2d)
    utils.convert_class_snn_to_yaml(net,'scnn_test')