# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:40:37 2025

@author: mrsco
"""

import torch    # PyTorch, what we are using for numerical processing on GPU
import time     # timing code
from tqdm.notebook import tqdm  # progress bar for loops
import matplotlib.pyplot as plt # plotting library
import src.utils as utils   # custom utilities for this project
import src.models as models # custom models for this project
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

print('Imports done')

config_dirs = {
    'path_res': '../results/torch_data/',      # directory to store results
    'path_comp': '../data/Completeness_783.csv',    # csv of the complete list of FlyWire neurons
    'path_conn': '../data/Connectivity_783.parquet', # connectivity data
    'path_wt': '../data', # weight directory
}

flyid2name = { }
#P9_oDN1 corresponds to forward velocity
P9_oDN1_left = 720575940626730883
P9_oDN1_right = 720575940620300308
P9_left = 720575940627652358
P9_right =720575940635872101
#DNa01 and DNa02 correspond to turning
DNa01_right = 720575940627787609
DNa01_left = 720575940644438551
DNa02_right = 720575940629327659
DNa02_left = 720575940604737708
#MDN, "Moonwalker descending neurons" are backwards walking/escape/startle response
MDN_1, MDN_2, MDN_3, MDN_4 = 720575940616026939, 720575940631082808, 720575940640331472,720575940610236514
#Giant Fiber corresponds to escape
Giant_Fiber_1, Giant_Fiber_2 = 720575940622838154,720575940632499757
#MN9 corresponds to proboscis motor neuron, and corresponds to feeding.
MN9_left = 720575940660219265
MN9_right = 720575940618238523
#aDN1 correspond to antennal grooming
aDN1_right = 720575940616185531
aDN1_left = 720575940624319124


flyid2name[P9_oDN1_left]=       "P9_oDN1_left"
flyid2name[P9_oDN1_right]=      "P9_oDN1_right"
flyid2name[P9_left]=    "P9_left"
flyid2name[P9_right]=   "P9_right"
flyid2name[DNa01_right]=        "DNa01_right"
flyid2name[DNa01_left]= "DNa01_left"
flyid2name[DNa02_right]=        "DNa02_right"
flyid2name[DNa02_left]= "DNa02_left"
flyid2name[MDN_1]= "MDN_1"
flyid2name[MDN_2]= "MDN_2"
flyid2name[MDN_3]= "MDN_3"
flyid2name[MDN_4]= "MDN_4"
flyid2name[Giant_Fiber_1]= "Giant_Fiber_1"
flyid2name[Giant_Fiber_2]= "Giant_Fiber_2"
flyid2name[MN9_left]= "MN9_left"
flyid2name[MN9_right]= "MN9_right"
flyid2name[aDN1_right]= "aDN1_right"
flyid2name[aDN1_left]= "aDN1_left"


# Simulation
stim_rate = 100
P9s = [P9_left, P9_right]   # we will stimulate P9 neurons in this example
flyid2i, i2flyid = utils.get_hash_tables(config_dirs['path_comp'])

sim_params = {
    'dt': 0.1,               # time step (ms)
    't_sim': 1000.0,         # total simulation time (ms)
    'batch_size': 1,        # batch size
}
num_steps = int(sim_params['t_sim'] / sim_params['dt'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_params = {
    'tauSyn': 5.0,    # ms
    'tDelay': 1.8,    # ms
    'v0': -52.0,      # mV
    'vReset': -52.0,  # mV
    'vRest': -52.0,   # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,   # ms
    'tRefrac': 2.2,    # ms
    'scalePoisson': 250, # scaling factor for poisson input
    'wScale': 0.275,      # scaling factor for synaptic weights
}

##FLAG -- weights not found
weights = utils.get_weights(config_dirs['path_conn'], config_dirs['path_comp'], config_dirs['path_wt'])


num_neurons = weights.shape[0]
model = models.TorchModel(sim_params['batch_size'], num_neurons, sim_params['dt'], model_params, weights.to(device=device), device=device)


# store the numberic IDs of the P9 neurons
excitatory_neurons = [flyid2i[n] for n in P9s]
# create a rates tensor with zeros, and do so on the correct device
rates = torch.zeros(sim_params['batch_size'],len(flyid2i), device=device)
# set the rates of the excitatory neurons to our desired stimulation rate
rates[:,excitatory_neurons] = stim_rate

conductance, delay_buffer, spikes, v, refrac = model.state_init()

if device == 'cuda':
    mem_used_gb = utils.get_vram_usage(device)
    print(f'Used {mem_used_gb:.4f} GB')


times_list = []
idx_list = []
times_list.append(torch.tensor([], device=device))
idx_list.append(torch.tensor([], device=device))
with torch.no_grad():
    start = time.time()
    conductance, delay_buffer, spikes, v, refrac = model.state_init()
    for t in tqdm(range(num_steps), desc='Steps'):
        conductance, delay_buffer, spikes, v, refrac = model(rates, conductance, delay_buffer, spikes, v, refrac)
        times_list[0], idx_list[0] = utils.get_spike_times(spikes[0,:], t, sim_params['dt'], times_list[0], idx_list[0])
    end = time.time()
print(f'Simulation time: {end - start:.2f} seconds')

time_per_step = (end - start)/num_steps*1000
print(f'Time per step: {time_per_step:.4f} ms')
print(f"Scale: {sim_params['dt']/time_per_step:.2f}x real-time simulation")


plt.figure()
plt.scatter(times_list[0].cpu(), idx_list[0].cpu(), s=100, marker='|')

if device == 'cuda':
    mem_used_gb = utils.get_vram_usage(device)
    print(f'Used {mem_used_gb:.4f} GB')