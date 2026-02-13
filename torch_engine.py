"""
PyTorch-based simulation engine using nourse_model.
Provides same API as sim_engine.py but uses TorchModel instead of CUDA kernels.
"""
import os
import sys
import numpy as np
import torch

# Add nourse_model to path
current_dir = os.path.dirname(__file__)
nourse_path = os.path.join(current_dir, 'nourse_model', 'src')
sys.path.insert(0, nourse_path)

import models as nourse_models
import utils as nourse_utils


class TorchSimEngine:
    """PyTorch-based connectome simulator using nourse_model."""
    
    def __init__(self, dt=0.1, device=None):
        """
        Initialize PyTorch simulation engine.
        
        Args:
            dt: Timestep in milliseconds (default: 0.1)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.dt = dt
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing TorchSimEngine (device: {self.device}, dt: {dt} ms)...")
        
        # Load connectome
        data_dir = os.path.join(current_dir, 'nourse_model', 'data')
        path_comp = os.path.join(data_dir, 'Completeness_783.csv')
        path_conn = os.path.join(data_dir, 'Connectivity_783.parquet')
        path_wt = data_dir
        
        # Get neuron ID mappings
        self.flyid2i, self.i2flyid = nourse_utils.get_hash_tables(path_comp)
        self.n_neurons = len(self.flyid2i)
        
        # Load weights
        weights = nourse_utils.get_weights(path_conn, path_comp, path_wt)
        
        # Model parameters (validated from walkingFly_Nourse.py)
        model_params = {
            'tauSyn': 5.0,        # Synaptic time constant (ms)
            'tDelay': 1.8,        # Synaptic delay (ms)
            'v0': -52.0,          # Initial voltage (mV)
            'vReset': -52.0,      # Reset voltage (mV)
            'vRest': -52.0,       # Rest voltage (mV)
            'vThreshold': -45.0,  # Spike threshold (mV)
            'tauMem': 20.0,       # Membrane time constant (ms)
            'tRefrac': 2.2,       # Refractory period (ms)
            'scalePoisson': 250,  # Poisson spike scaling
            'wScale': 0.275,      # Weight scaling factor
        }
        
        # Create the model
        self.model = nourse_models.TorchModel(
            batch=1,
            size=self.n_neurons,
            dt=self.dt,
            params=model_params,
            weights=weights.to(device=self.device),
            device=self.device
        )
        
        # Initialize state
        self._reset_state()
        
        # Tracking
        self._tracked_indices = []
        self._tracked_root_ids = []
        self._spike_counts = None
        
        # Stimulus rates
        self.rates = torch.zeros((1, self.n_neurons), device=self.device)
        
        print(f"Loaded connectome with {self.n_neurons} neurons")
    
    def _reset_state(self):
        """Reset simulation state to initial conditions."""
        self.conductance, self.delay_buffer, self.spikes, self.v, self.refrac = self.model.state_init()
        self.step_count = 0
    
    def _convert_ids_to_indices(self, neuron_ids):
        """
        Convert FlyWire root IDs or indices to neuron indices.
        Smart detection: if ID > n_neurons, treat as root ID, else as index.
        
        Args:
            neuron_ids: List of neuron identifiers (root IDs or indices, as int or string)
            
        Returns:
            Tuple of (indices, original_ids)
        """
        indices = []
        original_ids = []
        
        for nid in neuron_ids:
            # Convert to int if string (preserves precision through conversion)
            nid_int = int(nid)
            original_ids.append(nid_int)
            
            # If ID is larger than neuron count, treat as FlyWire root ID
            if nid_int > self.n_neurons:
                if nid_int in self.flyid2i:
                    indices.append(self.flyid2i[nid_int])
                else:
                    print(f"Warning: FlyWire ID {nid_int} not found in connectome")
            else:
                # Treat as direct index
                indices.append(nid_int)
        
        return indices, original_ids
    
    def inject_stimulus_by_rate(self, neuron_rates):
        """
        Set Poisson stimulus rates for specified neurons.
        
        Args:
            neuron_rates: List of dicts with 'id' (neuron ID, int or string) and 'rate' (Hz)
        """
        # Reset rates
        self.rates.zero_()
        
        for neuron_rate in neuron_rates:
            nid = int(neuron_rate['id'])  # Convert to int if string
            rate_hz = neuron_rate['rate']
            
            # Convert ID to index
            if nid > self.n_neurons and nid in self.flyid2i:
                idx = self.flyid2i[nid]
            else:
                idx = nid
            
            # Set rate (directly in Hz)
            self.rates[0, idx] = rate_hz
        
        print(f"Configured stimulus for {len(neuron_rates)} neurons")
    
    def set_tracked_neurons(self, neuron_ids):
        """
        Set which neurons to track for statistics.
        
        Args:
            neuron_ids: List of neuron IDs (root IDs or indices)
        """
        self._tracked_indices, self._tracked_root_ids = self._convert_ids_to_indices(neuron_ids)
        self._spike_counts = torch.zeros(len(self._tracked_indices), device=self.device)
        
        print(f"Tracking {len(self._tracked_indices)} neurons")
    
    def step(self, n_steps=1):
        """
        Run simulation for n steps.
        
        Args:
            n_steps: Number of timesteps to simulate
        """
        with torch.no_grad():
            for _ in range(n_steps):
                # Update model
                self.conductance, self.delay_buffer, self.spikes, self.v, self.refrac = self.model(
                    self.rates, self.conductance, self.delay_buffer, 
                    self.spikes, self.v, self.refrac
                )
                
                # Track spikes for monitored neurons
                if self._spike_counts is not None:
                    spikes_flat = self.spikes[0, :]  # Extract batch dimension
                    for i, neuron_idx in enumerate(self._tracked_indices):
                        if spikes_flat[neuron_idx] > 0:
                            self._spike_counts[i] += 1
                
                self.step_count += 1
    
    def get_tracked_neuron_stats(self, n_steps):
        """
        Get statistics for tracked neurons.
        
        Args:
            n_steps: Total number of steps simulated
            
        Returns:
            Dict with neuron_ids, spike_counts, firing_rates, and summary stats
        """
        if self._spike_counts is None:
            return {
                'neuron_ids': [],
                'spike_counts': [],
                'firing_rates': [],
                'mean_firing_rate': 0.0,
                'std_firing_rate': 0.0,
                'min_firing_rate': 0.0,
                'max_firing_rate': 0.0,
            }
        
        # Convert spike counts to CPU numpy
        spike_counts = self._spike_counts.cpu().numpy()
        
        # Calculate firing rates (spikes per timestep)
        firing_rates = spike_counts / n_steps if n_steps > 0 else np.zeros_like(spike_counts)
        
        return {
            'neuron_ids': self._tracked_root_ids.copy(),
            'spike_counts': spike_counts.tolist(),
            'firing_rates': firing_rates.tolist(),
            'mean_firing_rate': float(firing_rates.mean()) if len(firing_rates) > 0 else 0.0,
            'std_firing_rate': float(firing_rates.std()) if len(firing_rates) > 0 else 0.0,
            'min_firing_rate': float(firing_rates.min()) if len(firing_rates) > 0 else 0.0,
            'max_firing_rate': float(firing_rates.max()) if len(firing_rates) > 0 else 0.0,
        }
    
    def reset(self):
        """Reset simulation to initial state."""
        self._reset_state()
        self.rates.zero_()
        if self._spike_counts is not None:
            self._spike_counts.zero_()
        print("Simulation reset")
