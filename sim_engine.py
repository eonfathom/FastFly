"""
SimEngine — importable wrapper around the FlyWire CuPy simulator.

Loads neuron_annotations.npz (from download_metadata.py) for biologically
meaningful stimuli, heatmap groups, 3D positions, and motor neuron detail.

Usage (standalone test):
    python sim_engine.py                        # synthetic data
    python sim_engine.py --data flywire_v783.bin
"""

import base64
import os
import time
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed.  Run:  pip install cupy-cuda12x")
    sys.exit(1)

from flywire_sim import (CUDA_KERNELS, compile_kernels, load_connectome_binary,
                         generate_synthetic, quantize_weights_int8)

ANNOTATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "neuron_annotations.npz")


class SimEngine:
    """GPU-accelerated LIF simulator wrapping CuPy CUDA kernels."""

    def __init__(self, data_file=None, seed=42, dt=0.1):
        """Initialize the simulation engine.
        
        Args:
            data_file: Optional binary connectome file path
            seed: Random seed
            dt: Simulation timestep duration in milliseconds (default: 0.1 ms)
        """
        if data_file:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                load_connectome_binary(data_file)
        else:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                generate_synthetic(seed=seed)

        self.kernels = compile_kernels()
        self.seed = seed
        self.current_step = 0
        self.dt = dt  # Simulation timestep in milliseconds

        # GPU arrays — connectivity
        self.d_offsets = cp.asarray(offsets)
        self.d_targets = cp.asarray(targets)
        w_int8, w_scales = quantize_weights_int8(weights, offsets, self.n_neurons)
        self.d_weights = cp.asarray(w_int8)
        self.d_weight_scales = cp.asarray(w_scales)

        # GPU arrays — neuron state
        rng = cp.random.default_rng(seed)
        self.d_voltage = rng.uniform(0.0, 0.9, self.n_neurons).astype(cp.float32)
        self.d_current = cp.zeros(self.n_neurons, dtype=cp.float32)

        # Spike bookkeeping
        self.spike_words = (self.n_neurons + 31) // 32
        self.d_spike_bits = cp.zeros(self.spike_words, dtype=cp.uint32)
        self.d_spike_idx  = cp.zeros(self.n_neurons, dtype=cp.uint32)
        self.d_num_spikes = cp.zeros(1, dtype=cp.uint32)

        # LIF parameters
        self.tau_decay   = np.float32(0.9)
        self.v_threshold = np.float32(1.0)
        self.v_reset     = np.float32(0.0)
        self.noise_amp   = np.float32(0.4)

        # Launch config
        self.BLOCK = 256
        self.PROP_BLOCK = 128
        self.MAX_PROP_BLOCKS = 2048
        self.neuron_blocks  = (self.n_neurons + self.BLOCK - 1) // self.BLOCK
        self.compact_blocks = (self.spike_words + self.BLOCK - 1) // self.BLOCK

        # Stimulus state
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0
        
        # Rate-based stimulus (neuron_id -> firing_rate)
        self._rate_stim_neurons = None
        self._rate_stim_rates = None

        # Tracked neurons for custom I/O analysis
        self._tracked_neurons = None
        self._tracked_spike_counts = None

        # Last-step spike indices (for 3D viz)
        self._last_spike_indices = np.array([], dtype=np.int32)

        # GPU accumulators for sync-free counting
        self.d_total_spikes = cp.zeros(1, dtype=cp.uint64)

        # Feature toggles (can be set at runtime)
        self.send_active_indices = True
        self.send_group_rates = True
        self.send_motor_rates = True
        self.active_indices_interval = 3  # only transfer every Nth batch
        self._batch_counter = 0

        # Load annotations
        self._load_annotations()

        # Metrics history
        self.group_rates_history = []

    def _load_annotations(self):
        """Load neuron_annotations.npz for biological groups, stimuli, positions."""
        if os.path.exists(ANNOTATIONS_FILE):
            print(f"Loading neuron annotations from {ANNOTATIONS_FILE}...")
            data = np.load(ANNOTATIONS_FILE, allow_pickle=True)

            # Root IDs
            self._root_ids = data['root_ids'].astype(np.int64) if 'root_ids' in data else None
            
            # Create root_id -> index mapping for FlyWire ID lookups
            if self._root_ids is not None:
                self._root_id_to_index = {int(rid): idx for idx, rid in enumerate(self._root_ids)}
            else:
                self._root_id_to_index = None

            # 3D positions (normalized to [-1,1])
            if 'pos_x' in data:
                self._positions = np.stack([
                    data['pos_x'], data['pos_y'], data['pos_z']
                ], axis=1).astype(np.float32)  # [N, 3]
                print(f"  3D positions loaded: {self._positions.shape}")
            else:
                self._positions = None

            # Super class per neuron (for coloring in 3D)
            self._super_class = data.get('super_class', None)

            # Stimuli
            self._stimuli = {}
            stim_names = list(data['stim_names'])
            for name in stim_names:
                safe = 'stim_' + name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                if safe in data:
                    self._stimuli[name] = data[safe].astype(np.int32)

            # Heatmap groups
            self.group_labels = list(data['group_names'])
            self.num_groups = len(self.group_labels)
            self._group_indices = []
            for name in self.group_labels:
                self._group_indices.append(data['group_' + name].astype(np.int32))

            self._neuron_to_group = cp.full(self.n_neurons, -1, dtype=cp.int32)
            for g, indices in enumerate(self._group_indices):
                self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = g

            # Body sensory groups
            self._body_sensory = {}
            if 'body_sensory_names' in data:
                for name in data['body_sensory_names']:
                    key = 'bsens_' + name
                    if key in data:
                        self._body_sensory[str(name)] = data[key].astype(np.int32)

            # Body motor groups
            self._body_motor = {}
            if 'body_motor_names' in data:
                motor_names = [str(n) for n in data['body_motor_names']]
                for name in motor_names:
                    key = 'bmotor_' + name
                    if key in data:
                        self._body_motor[name] = data[key].astype(np.int32)

            self._motor_group_names = list(self._body_motor.keys())
            self._num_motor_groups = len(self._motor_group_names)
            self._neuron_to_motor = cp.full(self.n_neurons, -1, dtype=cp.int32)
            for g, name in enumerate(self._motor_group_names):
                indices = self._body_motor[name]
                self._neuron_to_motor[cp.asarray(indices.astype(np.int64))] = g

            self._use_annotations = True
            print(f"  {len(self._stimuli)} stimuli, {self.num_groups} heatmap groups")
            print(f"  {len(self._body_sensory)} body sensory, {len(self._body_motor)} body motor")

        else:
            print(f"No annotation file found ({ANNOTATIONS_FILE})")
            print("  Run download_metadata.py for biological annotations.")
            self._root_ids = None
            self._root_id_to_index = None
            self._positions = None
            self._super_class = None
            self._setup_fallback_groups()
            self._use_annotations = False

    def _setup_fallback_groups(self):
        """Fallback: equal-size index-range groups."""
        self.num_groups = 20
        group_size = self.n_neurons // self.num_groups
        self.group_labels = [f"Group {i}" for i in range(self.num_groups)]
        self._group_indices = []
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else self.n_neurons
            self._group_indices.append(np.arange(start, end, dtype=np.int32))

        self._neuron_to_group = cp.full(self.n_neurons, -1, dtype=cp.int32)
        for g, indices in enumerate(self._group_indices):
            self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = g

        self._stimuli = {
            "Neurons 0-1000": np.arange(0, min(1000, self.n_neurons), dtype=np.int32),
        }
        self._body_sensory = {}
        self._body_motor = {}
        self._motor_group_names = []
        self._num_motor_groups = 0
        self._neuron_to_motor = cp.full(self.n_neurons, -1, dtype=cp.int32)

    def _convert_ids_to_indices(self, neuron_ids):
        """Convert FlyWire root IDs to neuron indices.
        
        Args:
            neuron_ids: List of neuron IDs (can be root_ids or indices)
            
        Returns:
            numpy array of neuron indices
        """
        if self._root_id_to_index is None:
            # No mapping available, assume IDs are already indices
            return np.array(neuron_ids, dtype=np.int64)
        
        indices = []
        for nid in neuron_ids:
            nid_int = int(nid)
            # If it's a large number, assume it's a root_id
            if nid_int > self.n_neurons:
                if nid_int in self._root_id_to_index:
                    indices.append(self._root_id_to_index[nid_int])
                else:
                    raise ValueError(f"FlyWire root ID {nid_int} not found in connectome")
            else:
                # Small number, assume it's already an index
                indices.append(nid_int)
        
        return np.array(indices, dtype=np.int64)

    def inject_stimulus(self, neuron_indices, amplitude=0.5):
        """Inject constant current to specified neurons.
        
        Args:
            neuron_indices: List of neuron IDs
            amplitude: Current injection amplitude (0.0-1.0+)
        """
        self._stimulus_indices = cp.asarray(np.array(neuron_indices, dtype=np.int64))
        self._stimulus_amplitude = float(amplitude)
        # Clear rate-based stimulus
        self._rate_stim_neurons = None
        self._rate_stim_rates = None

    def inject_stimulus_by_rate(self, neuron_rates):
        """Inject stimulus by forcing neurons to spike at specified rates.
        
        Args:
            neuron_rates: List of dicts [{"id": neuron_id, "rate": firing_rate_Hz}, ...]
                         where neuron_id can be FlyWire root ID or neuron index,
                         and firing_rate_Hz is the desired firing rate in Hz (spikes/second)
        """
        if not neuron_rates:
            self._rate_stim_neurons = None
            self._rate_stim_rates = None
            return
        
        neuron_ids = [nr["id"] for nr in neuron_rates]
        rates_hz = [nr["rate"] for nr in neuron_rates]
        
        # Convert FlyWire IDs to indices
        neuron_indices = self._convert_ids_to_indices(neuron_ids)
        
        # Convert Hz to probability per timestep: prob = rate_Hz * (dt_ms / 1000)
        prob_scale = self.dt / 1000.0
        probabilities = [rate_hz * prob_scale for rate_hz in rates_hz]
        
        self._rate_stim_neurons = cp.asarray(neuron_indices)
        self._rate_stim_rates = cp.asarray(np.array(probabilities, dtype=np.float32))
        
        # Clear current-based stimulus
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0

    def clear_stimulus(self):
        """Clear all stimulus (both current-based and rate-based)."""
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0
        self._rate_stim_neurons = None
        self._rate_stim_rates = None

    def set_noise_amp(self, value):
        self.noise_amp = np.float32(value)

    def set_tracked_neurons(self, neuron_ids):
        """Set specific neurons to track for custom I/O analysis.
        
        Args:
            neuron_ids: List of neuron IDs (can be FlyWire root IDs or indices)
        """
        if neuron_ids is None or len(neuron_ids) == 0:
            self._tracked_neurons = None
            self._tracked_spike_counts = None
            self._tracked_neuron_root_ids = None
        else:
            # Convert to indices
            neuron_indices = self._convert_ids_to_indices(neuron_ids)
            self._tracked_neurons = cp.asarray(neuron_indices)
            self._tracked_spike_counts = cp.zeros(len(neuron_indices), dtype=cp.uint64)
            # Store original IDs for reporting
            self._tracked_neuron_root_ids = [int(nid) for nid in neuron_ids]

    def clear_tracked_neurons(self):
        """Clear tracked neurons."""
        self._tracked_neurons = None
        self._tracked_spike_counts = None
        self._tracked_neuron_root_ids = None

    def get_tracked_neuron_stats(self, n_steps):
        """Get firing rate statistics for tracked neurons.
        
        Args:
            n_steps: Number of timesteps that were run
            
        Returns:
            dict with per-neuron stats (with root IDs if available), or None if no neurons tracked
        """
        if self._tracked_neurons is None:
            return None
        
        spike_counts = self._tracked_spike_counts.get()
        neuron_indices = self._tracked_neurons.get()
        
        firing_rates = spike_counts / n_steps if n_steps > 0 else spike_counts * 0
        
        # Use original root IDs if we have them, otherwise use indices
        if hasattr(self, '_tracked_neuron_root_ids') and self._tracked_neuron_root_ids:
            neuron_ids = self._tracked_neuron_root_ids
        else:
            neuron_ids = neuron_indices.tolist()
        
        return {
            "neuron_ids": neuron_ids,
            "spike_counts": spike_counts.tolist(),
            "firing_rates": firing_rates.tolist(),
            "mean_firing_rate": float(firing_rates.mean()) if len(firing_rates) > 0 else 0.0,
            "std_firing_rate": float(firing_rates.std()) if len(firing_rates) > 0 else 0.0,
            "min_firing_rate": float(firing_rates.min()) if len(firing_rates) > 0 else 0.0,
            "max_firing_rate": float(firing_rates.max()) if len(firing_rates) > 0 else 0.0,
        }

    def reset_tracked_spike_counts(self):
        """Reset spike counts for tracked neurons."""
        if self._tracked_spike_counts is not None:
            self._tracked_spike_counts.fill(0)

    def step(self, n=50):
        """Run n timesteps and return a metrics dict.

        ZERO per-substep GPU→CPU syncs:
        - propagate_v2 reads d_num_spikes from device memory (no CPU readback)
        - count_spikes kernel counts groups/motors/total from spike_bits on GPU
        - compact kernel still runs (needed for propagate's spike_idx array)
        - Single sync at batch end to transfer results to CPU
        - active_indices only transferred when send_active_indices is True
        """
        t_start = time.perf_counter()

        # GPU accumulators — zeroed once, accumulated across all substeps (uint64 for atomicAdd)
        d_group_counts = cp.zeros(self.num_groups, dtype=cp.uint64)
        d_motor_counts = cp.zeros(max(self._num_motor_groups, 1), dtype=cp.uint64)
        d_total_spikes = self.d_total_spikes
        d_total_spikes.fill(0)

        # Local refs to avoid Python attribute lookups in inner loop
        d_current = self.d_current
        d_voltage = self.d_voltage
        d_spike_bits = self.d_spike_bits
        d_spike_idx = self.d_spike_idx
        d_num_spikes = self.d_num_spikes
        d_offsets = self.d_offsets
        d_targets = self.d_targets
        d_weights = self.d_weights
        d_weight_scales = self.d_weight_scales
        neuron_to_group = self._neuron_to_group
        neuron_to_motor = self._neuron_to_motor
        k_update_with_noise = self.kernels["update_with_noise"]
        k_compact = self.kernels["compact"]
        k_propagate_v2 = self.kernels["propagate_v2"]
        k_count = self.kernels["count_spikes"]
        neuron_blocks = self.neuron_blocks
        compact_blocks = self.compact_blocks
        BLOCK = self.BLOCK
        PROP_BLOCK = self.PROP_BLOCK
        MAX_PROP_BLOCKS = self.MAX_PROP_BLOCKS
        n_neurons_i32 = np.int32(self.n_neurons)
        spike_words_i32 = np.int32(self.spike_words)
        stim_indices = self._stimulus_indices
        stim_amp = self._stimulus_amplitude
        rate_stim_neurons = self._rate_stim_neurons
        rate_stim_rates = self._rate_stim_rates

        for sub in range(n):
            d_num_spikes.fill(0)

            # Current-based stimulus
            if stim_indices is not None:
                d_current[stim_indices] += stim_amp

            k_update_with_noise(
                (neuron_blocks,), (BLOCK,),
                (d_voltage, d_current, d_spike_bits,
                 n_neurons_i32, spike_words_i32,
                 self.tau_decay, self.v_threshold, self.v_reset,
                 np.uint32(self.seed), np.uint32(self.current_step),
                 self.noise_amp))

            # Rate-based stimulus - inject Poisson current (biologically realistic)
            # Instead of forcing spikes, generate Poisson input and inject as current
            if rate_stim_neurons is not None:
                rng = cp.random.default_rng(self.seed + self.current_step)
                rand_vals = rng.random(len(rate_stim_neurons), dtype=cp.float32)
                # Poisson spike generation: prob = rate * dt/1000
                # If spike occurs, inject scaled current (like PyTorch scalePoisson=250)
                current_scale = 250.0  # Match PyTorch scalePoisson parameter
                for i in range(len(rate_stim_neurons)):
                    if rand_vals[i] < rate_stim_rates[i]:
                        nid = int(rate_stim_neurons[i])
                        # Inject current instead of forcing spike
                        d_current[nid] += current_scale

            k_compact(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, d_spike_idx, d_num_spikes,
                 spike_words_i32, n_neurons_i32))

            # Count group/motor/total spikes from spike_bits — pure GPU, no sync
            k_count(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, neuron_to_group, d_group_counts,
                 neuron_to_motor, d_motor_counts, d_total_spikes,
                 spike_words_i32, n_neurons_i32))

            # Track specific neurons if requested
            if self._tracked_neurons is not None:
                for i, nid in enumerate(self._tracked_neurons):
                    word_idx = nid // 32
                    bit_idx = nid % 32
                    if d_spike_bits[word_idx] & (1 << bit_idx):
                        self._tracked_spike_counts[i] += 1

            # Propagate v2 — reads d_num_spikes from device memory, no CPU sync
            k_propagate_v2(
                (MAX_PROP_BLOCKS,), (PROP_BLOCK,),
                (d_spike_idx, d_num_spikes,
                 d_offsets, d_targets, d_weights, d_weight_scales, d_current))

            self.current_step += 1

        # === Single batch-end sync — all GPU work done ===
        cp.cuda.Stream.null.synchronize()

        total_spikes = int(d_total_spikes[0])
        t_elapsed = time.perf_counter() - t_start
        firing_rate = total_spikes / (n * self.n_neurons) if self.n_neurons > 0 else 0
        steps_per_sec = n / t_elapsed if t_elapsed > 0 else 0

        result = {
            "step": self.current_step,
            "spike_count": total_spikes,
            "firing_rate": round(firing_rate, 6),
            "mean_voltage": round(float(d_voltage.mean()), 4),
            "steps_per_sec": round(steps_per_sec, 1),
        }

        # Group rates (for heatmap) — only compute if enabled
        if self.send_group_rates:
            group_spike_counts = d_group_counts.get()
            group_rates = []
            for g in range(self.num_groups):
                group_n = len(self._group_indices[g])
                rate = float(group_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                group_rates.append(round(rate, 6))
            self.group_rates_history.append(group_rates)
            if len(self.group_rates_history) > 200:
                self.group_rates_history = self.group_rates_history[-200:]
            result["group_rates"] = group_rates

        # Motor rates — only compute if enabled
        if self.send_motor_rates:
            motor_spike_counts = d_motor_counts.get()
            motor_rates = {}
            for g, name in enumerate(self._motor_group_names):
                group_n = len(self._body_motor[name])
                rate = float(motor_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                motor_rates[name] = round(rate, 6)
            result["motor_rates"] = motor_rates

        # Active indices (for 3D viz) — only transfer every Nth batch
        self._batch_counter += 1
        if self.send_active_indices and self._batch_counter % self.active_indices_interval == 0:
            num_last = int(d_num_spikes[0])
            if num_last > 0:
                self._last_spike_indices = d_spike_idx[:num_last].get().astype(np.int32)
            else:
                self._last_spike_indices = np.array([], dtype=np.int32)
            result["active_indices"] = self._last_spike_indices.tolist()

        return result

    # --- Data accessors ---

    def get_predefined_stimuli(self):
        return list(self._stimuli.keys())

    def get_body_info(self):
        sensory = {k: len(v) for k, v in self._body_sensory.items()}
        motor = {k: len(v) for k, v in self._body_motor.items()}
        return {"sensory": sensory, "motor": motor}

    def get_positions_b64(self):
        """Return neuron positions as base64-encoded float32 array [N*3]."""
        if self._positions is not None:
            return base64.b64encode(self._positions.tobytes()).decode('ascii')
        return None

    def get_neuron_classes(self):
        """Return super_class per neuron for 3D coloring."""
        if self._super_class is not None:
            # Encode as int: unique classes -> color indices
            unique = sorted(set(self._super_class))
            class_to_id = {c: i for i, c in enumerate(unique)}
            ids = np.array([class_to_id.get(c, 0) for c in self._super_class],
                           dtype=np.uint8)
            return {
                "labels": unique,
                "ids_b64": base64.b64encode(ids.tobytes()).decode('ascii')
            }
        return None

    def get_motor_detail(self, group_name):
        """Return detail for a motor group: neuron indices, root_ids, active status."""
        if group_name not in self._body_motor:
            return None
        indices = self._body_motor[group_name]
        active_set = set(self._last_spike_indices.tolist())
        neurons = []
        for idx in indices:
            idx = int(idx)
            rid = int(self._root_ids[idx]) if self._root_ids is not None else idx
            neurons.append({
                "index": idx,
                "root_id": rid,
                "active": idx in active_set,
            })
        return {"group": group_name, "neurons": neurons}

    def apply_predefined_stimulus(self, name, amplitude=None):
        if name not in self._stimuli:
            return False
        indices = self._stimuli[name]
        amp = amplitude if amplitude is not None else 0.5
        self.inject_stimulus(indices, amp)
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Binary connectome file")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    engine = SimEngine(data_file=args.data)
    print(f"\nSimEngine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses")
    print(f"Stimuli: {engine.get_predefined_stimuli()}")
    print(f"Groups:  {engine.group_labels}")
    print(f"Positions: {'yes' if engine._positions is not None else 'no'}")
    print(f"Running {args.steps} steps in batches of {args.batch}...\n")

    for i in range(0, args.steps, args.batch):
        metrics = engine.step(n=args.batch)
        print(f"  Step {metrics['step']:>6d}  "
              f"spikes={metrics['spike_count']:>6d}  "
              f"rate={metrics['firing_rate']*100:>5.2f}%  "
              f"V_mean={metrics['mean_voltage']:.3f}  "
              f"active_3d={len(metrics.get('active_indices', []))}  "
              f"steps/s={metrics['steps_per_sec']:.0f}")

    print("\nDone.")
