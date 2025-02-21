import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Citation: Network definition from neurobench primate_reaching.py example
## Define model ##
class SNN2(nn.Module):

    def __init__(self, window=50, input_size=96, hidden_size=50, tau=0.96, p=0.3, device='cpu'):
        super().__init__()

        # self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.surrogate = surrogate.fast_sigmoid(slope=20)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False, device=device)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)

        self.lif1 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='zero')
        self.lif_out = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='none')

        self.dropout = nn.Dropout(p)
        self.mem1, self.mem2 = None, None

        # self.register_buffer('inp', torch.zeros(window, self.input_size))

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif_out.init_leaky()

    def single_forward(self, x):
        x = x.squeeze() # convert shape (1, input_dim) to (input_dim)
        cur1 = self.dropout(self.fc1(x))
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        cur2 = self.fc_out(spk1)
        _, self.mem2 = self.lif_out(cur2, self.mem2)

        return self.mem2.clone()

    def forward(self, x):
        # here x is expected to be shape (len_series, 1, input_dim)
        predictions = []

        for sample in range(x.shape[0]):
            predictions.append(self.single_forward(x[sample, ...]))

        predictions = torch.stack(predictions)
        return predictions


if __name__ == "__main__":
    
    # Create the model (SNN)
    # Initialize and load the network. The SNN2 model architecture is 
    # a very simple two-layer linear feedforward SNN, which is implemented using SNNTorch.
    
    # net = SNN2(input_size=dataset.input_feature_size) # maybe don't pass this in and use defaults? getting the dataset will be tuff...
    net = SNN2(input_size = 96) # Pretty sure this is 96 based on the dataloader stuff in the ipynb, different values don't work.
    
    # Load the pre-trained weights
    filename = "indy_20160622_01" # specify a dataset (TODO: verify this is the right one)
    model_path = r"/home/ala4225/neurobench/neurobench/examples/primate_reaching/model_data"
    model_path = model_path + "/SNN2_{}.pt"
    net.load_state_dict(torch.load(model_path.format(filename), map_location=torch.device('cpu'), weights_only = True) 
                        ['model_state_dict'], strict=False)

    layer_list = [net.dropout, net.fc1, net.lif1, net.fc_out, net.lif_out]

    # TODO: validate the network with the gsc example

    