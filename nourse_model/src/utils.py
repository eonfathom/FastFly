"""
Copyright (c) Eon Systems, 2025.
All rights reserved. Unauthorized copying, distribution, or modification
of this file, in whole or in part, is strictly prohibited.
"""
import torch
import pandas as pd
import pickle
import numpy as np

def get_spike_times(spikes, i, dt, spike_times, spike_idx):
    """
    Given raw spiking data, where every neuron is represented as 0 or 1,
    get the spike times and indices of neurons that spiked.

    Args:
        spikes (tensor): Tensor with spikes
        i (int): Current step index
        dt (float): Simulation time step (in ms)
        spike_times (tensor): Tensor of firing times
        spike_idx (tensor): Tensor of neural indices that fired at firing times

    Returns:
        tensor: Updated tensor of firing times
        tensor: Updated tensor of neural indices that fired at firing times
    """
    idx = torch.nonzero(spikes).squeeze()
    times = torch.zeros_like(idx)+((i-1)*dt)
    spike_times = torch.hstack((spike_times, times))
    spike_idx = torch.hstack((spike_idx, idx))
    return spike_times, spike_idx

def get_hash_tables(path_comp):
    """
    Get mapping between flywire IDs and torch IDs

    Args:
        path_comp (str): Path to the completeness csv file

    Returns:
        dict: Mapping from flywire ID to torch ID
        dict: Mapping from torch ID to flywire ID
    """
    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: torch ID
    i2flyid = {j: i for i, j in flyid2i.items()} # torch ID: flywire ID
    return flyid2i, i2flyid

def get_weights(path_conn, path_comp, path_wt, csr=True):
    """
    Get weight matrix from connectivity data

    Args:
        path_conn (str): Path to connectivity parquet file
        path_comp (str): Path to completeness csv file
        path_wt (str): Path to weight matrix directory
        csr (bool, optional): If True, use CSR sparse formatting.
          If False, use COO. Defaults to True.

    Returns:
        sparse tensor: Sparse representation of weight matrix,
            either in CSR or COO format
    """
    data_conn = pd.read_parquet(path_conn)
    data_name = pd.read_csv(path_comp)
    num_neurons = data_name.shape[0]

    # Try to load existing weight matrix in COO format
    try:
        weight_coo = pickle.load(open(path_wt+'/weight_coo.pkl', 'rb'))
    except FileNotFoundError:
        print('Weights not found, constructing COO weight matrix')
        idx = [data_conn['Postsynaptic_Index'].to_list(), data_conn['Presynaptic_Index'].to_list()] # ordered pairs of (post, pre)
        val = data_conn['Excitatory x Connectivity'].to_list()  # weight values corresponding to ordered pairs
        weight_coo = torch.sparse_coo_tensor(idx, val, (num_neurons, num_neurons)).to(torch.float32)    # generate COO sparse tensor
        pickle.dump(weight_coo, open(path_wt+'/weight_coo.pkl', 'wb'))  # save to disk

    if csr:
        # Try to load existing weight matrix in CSR format
        try:
            weight_csr = pickle.load(open(path_wt+'/weight_csr.pkl', 'rb'))
        except FileNotFoundError:
            print('Weights not found, constructing CSR weight matrix')
            weight_csr = weight_coo.to_sparse_csr() # generate CSR sparse tensor from COO format
            pickle.dump(weight_csr, open(path_wt+'/weight_csr.pkl', 'wb'))  # save to disk
        return weight_csr
    else:
        return weight_coo

def get_vram_usage(device):
    """
    Get VRAM usage on a given device

    Args:
        device (str): Which CUDA device to check

    Returns:
        float: Memory used in GB
    """
    free, total = torch.cuda.mem_get_info(device)
    mem_used_gb = (total - free) / 1024 ** 3
    return mem_used_gb

def load_exps(l_pqt):
    '''Load simulation results from disk. From Shiu et al. 2024

    Parameters
    ----------
    l_pkl : list
        List of parquet files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pqt:
        # load metadata from pickle
        with open(p, 'rb') as f:
            df = pd.read_parquet(p)
            df.loc[:, 't'] = df.loc[:, 't'].astype(float)
            dfs.append(df)

    df = pd.concat(dfs)

    return df

def construct_dataframe(spike_times_list, spike_idx_list, i2flyid, exp_name, dt):
    """
    Given lists of spike times and spike indices for multiple trials,
    construct a pandas dataframe with all spike data. From Shiu et al. 2024

    Args:
        spike_times_list (list): List of spike times
        spike_idx_list (list): List of firing neuron indices
        i2flyid (dict): Hash table mapping torch IDs to flywire IDs
        exp_name (str): Experiment name
        dt (float): Simulation time step (in ms)

    Returns:
        dataframe: Pandas dataframe with all spike data
    """
    all_times = []
    all_idx = []
    all_trials = []
    for trial in range(len(spike_times_list)):
        spike_times = spike_times_list[trial].to('cpu')
        spike_idx = spike_idx_list[trial]
        nrun = []
        nrun.extend([trial]*spike_times.shape[0])
        all_times.extend(spike_times*dt)
        all_idx.extend(spike_idx)
        all_trials.extend(nrun)
    d = {
        't': [i.item() for i in all_times],
        'trial': all_trials,
        'flywire_id': [i2flyid[int(i)] for i in all_idx],
        'exp_name': exp_name
    }
    df = pd.DataFrame(d)
    return df

def get_rate(df, t_run, n_run, flyid2name=dict()):
    '''Calculate rate and standard deviation for all experiments
    in df. From Shiu et al. 2024

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    t_run : float
        Trial duration in seconds
    n_run : int
        Number of trials
    flyid2name : dict (optional)
        Mapping between flywire IDs and custom names

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = np.zeros(n_run)
            for t, df_t in df_f.groupby('trial'):
                r[int(t)] = len(df_t) / t_run

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std