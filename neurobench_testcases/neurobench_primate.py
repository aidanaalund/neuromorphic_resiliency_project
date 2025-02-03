from torch.utils.data import DataLoader, Subset
from neurobench.datasets import PrimateReaching
import yaml
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch as snn

from neurobench.models import SNNTorchModel
from neurobench.models import TorchModel

from neurobench.metrics.workload import (
    R2,
    ActivationSparsity,
    SynapticOperations,
    MembraneUpdates
)

from neurobench.metrics.static import (
    ConnectionSparsity,
    Footprint
)

from neurobench.benchmarks import Benchmark

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
    # The dataloader and preprocessor has been combined together into a single class
    data_dir = "/home/ala4225/nmr_project/neurobench_testcases/data" # data in repo root dir

    filename = "indy_20160622_01"

    dataset = PrimateReaching(file_path=data_dir, filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)
    
    net = SNN2(input_size = dataset.input_feature_size) # Pretty sure this is 96 based on the dataloader stuff in the ipynb, different values don't work.
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)
    # Load the pre-trained weights 
    # # specify a dataset (TODO: verify this is the right one)
    model_path = r"/home/ala4225/neurobench/neurobench/examples/primate_reaching/model_data"
    model_path = model_path + "/SNN2_{}.pt"
    net.load_state_dict(torch.load(model_path.format(filename), map_location=torch.device('cpu'), weights_only = True) 
                        ['model_state_dict'], strict=False)
    # Wrap our net in the SNNTorchModel wrapper
    net.reset()
    model = TorchModel(net) # using TorchModel instead of SNNTorchModel because the SNN iterates over dimension 0
    model.add_activation_module(snn.SpikingNeuron)

    preprocessors = []
    postprocessors = []

    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [R2, ActivationSparsity, SynapticOperations, MembraneUpdates]

    benchmark = Benchmark(model, test_set_loader,
                        preprocessors, postprocessors, [static_metrics, workload_metrics])

    results = benchmark.run() # this takes FOREVER
    #print(results)
    with open('primate_neurobench.yaml', 'w') as file:
        yaml.dump(results, file)
    # TODO: save the results in a file

