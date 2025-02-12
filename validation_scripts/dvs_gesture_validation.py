import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Citation: Network definition from neurobench dvs_gesture example
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        beta_1 = 0.9999903192467171
        beta_2 = 0.7291118090686332
        beta_3 = 0.9364650136740154
        beta_4 = 0.8348241794080301
        threshold_1 = 3.511291184386264
        threshold_2 = 3.494437965584431
        threshold_3 = 1.5986853560315544
        threshold_4 = 0.3641469130041378
        spike_grad = surrogate.atan()
        dropout = 0.5956071342984011
        
        # Initialize layers

        # torch.nn.Conv2d(in_channels, out_channels, 
        # kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        # bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv1 = nn.Conv2d(2, 16, 5, padding="same") # 2 is C, 16 is N, 5x5 filter with a stride of 1
        self.pool1 = nn.MaxPool2d(2) # 2x2 kernel with stride of 2 (kernel_size) (16 different 5x5 outputs that are made into a 2x2 by a maxpool filter)
        self.lif1 = snn.Leaky(beta=beta_1, threshold=threshold_1, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(16, 32, 5, padding="same")
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta_2, threshold=threshold_2, spike_grad=spike_grad, init_hidden=True)
        
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta_3, threshold=threshold_3, spike_grad=spike_grad, init_hidden=True)
        
        self.linear1 = nn.Linear(64*4*4, 11)
        self.dropout_4 = nn.Dropout(dropout) # Ignore since this is a function of traning?
        self.lif4 = snn.Leaky(beta=beta_4, threshold=threshold_4, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        # x is expected to be in shape (batch, channels, height, width) = (B, 2, 32, 32)
        
        # Layer 1 (conv and maxpooling are combined under the hood)
        y = self.conv1(x)
        y = self.pool1(y)
        spk1 = self.lif1(y)

        # Layer 2
        y = self.conv2(spk1)
        y = self.pool2(y)
        spk2 = self.lif2(y)

        # Layer 3
        y = self.conv3(spk2)
        y = self.pool3(y)
        spk3 = self.lif3(y)

        # Layer 4
        y = self.linear1(spk3.flatten(1))
        y = self.dropout_4(y)
        spk4, mem4 = self.lif4(y)

        return spk4, mem4

if __name__ == "__main__":
    # initialize network
    net = Net()
    
    # load pretrained weights (meaning dropout weights are already included)
    model_path = r"/home/ala4225/neurobench/neurobench/examples/dvs_gesture/model_data/dvs_gesture_snn"
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=True))

    print(net.state_dict().keys())