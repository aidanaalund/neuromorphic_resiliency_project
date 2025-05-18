import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import os
import sys

# import sanafe stuff
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, PROJECT_DIR)
import sanafe

ARCH_FILENAME = "loihi.yaml"
NETWORK_FILENAME = "dvs_gesture_32x32.net"
LOIHI_TIME_DATA_FILENAME = "loihi_gesture_32x32_time.csv"
LOIHI_ENERGY_DATA_FILENAME = "loihi_gesture_32x32_energy.csv"
SIM_TIME_DATA_FILENAME = "sim_gesture_32x32_time.csv"
SIM_ENERGY_DATA_FILENAME = "sim_gesture_32x32_energy.csv"

#NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32_apr03")
NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32_sep30")
#NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32")
DVS_RUN_DIR = os.path.join(PROJECT_DIR, "runs", "dvs")

ARCH_PATH = os.path.join(PROJECT_DIR, "arch", ARCH_FILENAME)
GENERATED_NETWORK_PATH = os.path.join(DVS_RUN_DIR, NETWORK_FILENAME)
#LOIHI_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, "loihi_gesture_32x32_apr03", LOIHI_TIME_DATA_FILENAME)
LOIHI_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_TIME_DATA_FILENAME)
LOIHI_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_ENERGY_DATA_FILENAME)
SIM_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_TIME_DATA_FILENAME)
SIM_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_ENERGY_DATA_FILENAME)



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



    # time step of spikes. Doesn't have to be spike perfect but get close. Convert to csv.
    
    # Run the snntorch

    # Run the sanafe stuff
    # look at dvs_gesture.py in sanafe's scripts folder
    
