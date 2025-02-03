import torch
# import the dataloader
from torch.utils.data import DataLoader

# import the dataset, preprocessors and postprocessors you want to use
from neurobench.datasets import SpeechCommands
from neurobench.processors.preprocessors import S2SPreProcessor
from neurobench.processors.postprocessors import ChooseMaxCount

# import the NeuroBench wrapper to wrap the snnTorch model
from neurobench.models import SNNTorchModel

# import metrics
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    ClassificationAccuracy,
    MembraneUpdates
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

# import the benchmark class
from neurobench.benchmarks import Benchmark

from torch import nn
import snntorch as snn
from snntorch import surrogate

import yaml

if __name__ == "__main__":

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
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )

    model_path = r"/home/ala4225/neurobench/neurobench/examples/gsc/model_data/s2s_gsc_snntorch"
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only = True))
    # Wrap our net in the SNNTorchModel wrapper
    model = SNNTorchModel(net)

    test_set = SpeechCommands(path="/home/ala4225/nmr_project/neurobench_testcases/data", subset="testing")
    test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

    preprocessors = [S2SPreProcessor(device='cpu')] # CHANGE
    postprocessors = [ChooseMaxCount()]

    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [ClassificationAccuracy, ParameterCount, ActivationSparsity, SynapticOperations, MembraneUpdates]

    benchmark = Benchmark(model, test_set_loader,
                        preprocessors, postprocessors, [static_metrics, workload_metrics])

    results = benchmark.run()
    #print(results)
    with open('gsc_neurobench.yaml', 'w') as file:
        yaml.dump(results, file)