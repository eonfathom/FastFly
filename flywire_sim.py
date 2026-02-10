"""
FlyWire Connectome GPU Simulator (Python + CuPy)

Same CUDA kernels as the .cu version, compiled at runtime via nvrtc.
No cl.exe / Visual Studio needed.

Usage:
  pip install cupy-cuda12x
  python flywire_sim.py --data flywire_v783.bin
  python flywire_sim.py                          # synthetic data fallback
"""

import argparse
import struct
import time
import sys
import numpy as np

try:
    import cupy as cp
    from cupyx.profiler import benchmark as cp_benchmark
except ImportError:
    print("ERROR: CuPy not installed.")
    print("Run:  pip install cupy-cuda12x")
    sys.exit(1)

# ================================================================
# CUDA Kernels (compiled at runtime via nvrtc - no cl.exe needed)
# ================================================================

CUDA_KERNELS = r"""
extern "C" {

// Noise injection - hash-based pseudo-random background current
__global__ void kernel_inject_noise(
    float* __restrict__ current,
    const unsigned int   seed,
    const unsigned int   step,
    const float          amplitude,
    const int            N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int h = i ^ (step * 2654435761u) ^ (seed * 1664525u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;

    float noise = amplitude * (((float)(h & 0xFFFF) / 32768.0f) - 1.0f);
    current[i] += noise;
}

// LIF neuron update + warp ballot spike detection
__global__ void kernel_update_neurons(
    float*        __restrict__ voltage,
    float*        __restrict__ current,
    unsigned int* __restrict__ spike_bits,
    const int                  N,
    const int                  spike_words,
    const float                tau_decay,
    const float                v_threshold,
    const float                v_reset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool spiked = false;
    if (i < N) {
        float v = voltage[i] * tau_decay + current[i];
        spiked = (v >= v_threshold);
        voltage[i] = spiked ? v_reset : v;
        current[i] = 0.0f;
    }

    unsigned int ballot = __ballot_sync(0xFFFFFFFF, spiked);
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        int word = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        if (word < spike_words)
            spike_bits[word] = ballot;
    }
}

// Stream compaction - extract spiking neuron indices
__global__ void kernel_compact_spikes(
    const unsigned int* __restrict__ spike_bits,
    unsigned int*       __restrict__ spike_idx,
    unsigned int*       __restrict__ num_spikes,
    const int                        num_words,
    const int                        N
) {
    __shared__ unsigned int s_offsets[256];
    __shared__ unsigned int block_base;

    int tid = threadIdx.x;
    int word_idx = blockIdx.x * blockDim.x + tid;

    unsigned int bits = 0;
    int count = 0;
    if (word_idx < num_words) {
        bits = spike_bits[word_idx];
        count = __popc(bits);
    }
    s_offsets[tid] = count;
    __syncthreads();

    // Inclusive prefix sum
    for (int stride = 1; stride < 256; stride <<= 1) {
        unsigned int val = 0;
        if (tid >= stride) val = s_offsets[tid - stride];
        __syncthreads();
        s_offsets[tid] += val;
        __syncthreads();
    }

    unsigned int block_total = s_offsets[255];
    unsigned int my_offset = (tid > 0) ? s_offsets[tid - 1] : 0;

    if (tid == 0 && block_total > 0) {
        block_base = atomicAdd(num_spikes, block_total);
    }
    __syncthreads();

    if (word_idx < num_words && bits != 0) {
        unsigned int base_neuron = word_idx * 32;
        unsigned int write_pos = block_base + my_offset;
        while (bits) {
            int b = __ffs(bits) - 1;
            unsigned int neuron = base_neuron + b;
            if (neuron < (unsigned int)N)
                spike_idx[write_pos++] = neuron;
            bits &= bits - 1;
        }
    }
}

// Spike propagation - push model, warp-per-spike, grid-stride
__global__ void kernel_propagate_spikes(
    const unsigned int* __restrict__ spike_idx,
    const unsigned int               num_spikes,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const float*        __restrict__ weights,
    float*              __restrict__ current
) {
    unsigned int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int num_warps = (gridDim.x * blockDim.x) >> 5;
    unsigned int lane = threadIdx.x & 31;

    for (unsigned int s = warp_id; s < num_spikes; s += num_warps) {
        unsigned int neuron    = spike_idx[s];
        unsigned int syn_start = offsets[neuron];
        unsigned int syn_end   = offsets[neuron + 1];

        for (unsigned int syn = syn_start + lane; syn < syn_end; syn += 32) {
            unsigned int t = targets[syn];
            float w = weights[syn];
            atomicAdd(&current[t], w);
        }
    }
}

// Spike propagation v2 - reads num_spikes from device pointer (no CPU sync needed)
__global__ void kernel_propagate_v2(
    const unsigned int* __restrict__ spike_idx,
    const unsigned int* __restrict__ p_num_spikes,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const float*        __restrict__ weights,
    float*              __restrict__ current
) {
    unsigned int num_spikes = *p_num_spikes;
    if (num_spikes == 0) return;

    unsigned int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int num_warps = (gridDim.x * blockDim.x) >> 5;
    unsigned int lane = threadIdx.x & 31;

    for (unsigned int s = warp_id; s < num_spikes; s += num_warps) {
        unsigned int neuron    = spike_idx[s];
        unsigned int syn_start = offsets[neuron];
        unsigned int syn_end   = offsets[neuron + 1];

        for (unsigned int syn = syn_start + lane; syn < syn_end; syn += 32) {
            unsigned int t = targets[syn];
            float w = weights[syn];
            atomicAdd(&current[t], w);
        }
    }
}

// Count group/motor spikes directly from spike_bits (no compact needed for counting)
__global__ void kernel_count_spikes(
    const unsigned int*  __restrict__ spike_bits,
    const int*           __restrict__ neuron_to_group,
    unsigned long long*  __restrict__ group_counts,
    const int*           __restrict__ neuron_to_motor,
    unsigned long long*  __restrict__ motor_counts,
    unsigned long long*  __restrict__ total_spikes,
    const int                         num_words,
    const int                         N
) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_idx >= num_words) return;
    unsigned int bits = spike_bits[word_idx];
    if (bits == 0) return;

    atomicAdd(total_spikes, (unsigned long long)__popc(bits));

    unsigned int base = (unsigned int)word_idx * 32u;
    while (bits) {
        int b = __ffs(bits) - 1;
        unsigned int neuron = base + (unsigned int)b;
        if (neuron < (unsigned int)N) {
            int g = neuron_to_group[neuron];
            if (g >= 0) atomicAdd(&group_counts[g], 1ULL);
            int m = neuron_to_motor[neuron];
            if (m >= 0) atomicAdd(&motor_counts[m], 1ULL);
        }
        bits &= bits - 1;
    }
}

}  // extern "C"
"""

# ================================================================
# Compile kernels
# ================================================================

def compile_kernels():
    module = cp.RawModule(code=CUDA_KERNELS, options=("--std=c++11",))
    return {
        "noise":       module.get_function("kernel_inject_noise"),
        "update":      module.get_function("kernel_update_neurons"),
        "compact":     module.get_function("kernel_compact_spikes"),
        "propagate":   module.get_function("kernel_propagate_spikes"),
        "propagate_v2": module.get_function("kernel_propagate_v2"),
        "count_spikes": module.get_function("kernel_count_spikes"),
    }

# ================================================================
# Load connectome from binary file
# ================================================================

def load_connectome_binary(filename):
    print(f"Loading connectome from '{filename}'...")

    with open(filename, "rb") as f:
        magic, version, n_neurons, n_synapses = struct.unpack("<IIII", f.read(16))

        if magic != 0x464C5957:
            print(f"ERROR: Invalid file magic 0x{magic:X}")
            sys.exit(1)

        print(f"  Neurons:  {n_neurons}")
        print(f"  Synapses: {n_synapses}")

        offsets = np.frombuffer(f.read((n_neurons + 1) * 4), dtype=np.uint32).copy()
        targets = np.frombuffer(f.read(n_synapses * 4), dtype=np.uint32).copy()
        weights = np.frombuffer(f.read(n_synapses * 4), dtype=np.float32).copy()

    degrees = np.diff(offsets)
    print(f"  Out-degree: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.1f}")

    n_exc = (weights[:min(n_synapses, 1000000)] > 0).sum()
    print(f"  Excitatory (sample): {100*n_exc/min(n_synapses,1000000):.1f}%")
    print(f"  Loaded successfully.\n")

    return n_neurons, n_synapses, offsets, targets, weights

# ================================================================
# Generate synthetic connectome
# ================================================================

def generate_synthetic(n_neurons=139255, n_synapses=54500000, seed=42):
    print(f"Generating synthetic connectome ({n_neurons} neurons, {n_synapses} synapses)...")
    rng = np.random.default_rng(seed)

    mean_deg = n_synapses / n_neurons
    log_mean = np.log(mean_deg) - 0.5 * np.log(5.0)
    log_std = np.sqrt(np.log(5.0))
    degrees = rng.lognormal(log_mean, log_std, n_neurons).astype(np.int64)
    degrees = np.clip(degrees, 1, n_neurons - 1)

    # Rescale to target
    scale = n_synapses / degrees.sum()
    degrees = np.maximum(1, np.round(degrees * scale)).astype(np.int64)
    diff = n_synapses - degrees.sum()
    if diff != 0:
        idx = rng.choice(n_neurons, abs(int(diff)), replace=True)
        degrees[idx] += np.sign(diff)
        degrees = np.maximum(1, degrees)

    actual_synapses = int(degrees.sum())
    offsets = np.zeros(n_neurons + 1, dtype=np.uint32)
    np.cumsum(degrees, out=offsets[1:])

    targets = rng.integers(0, n_neurons, actual_synapses, dtype=np.uint32)
    excitatory = rng.random(n_neurons) < 0.7
    weights = np.abs(rng.normal(0, 0.03, actual_synapses)).astype(np.float32) + 0.005
    # Apply Dale's law per neuron
    for i in range(n_neurons):
        if not excitatory[i]:
            weights[offsets[i]:offsets[i+1]] *= -1

    print(f"  Generated {actual_synapses} synapses")
    print(f"  Degree: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.1f}\n")
    return n_neurons, actual_synapses, offsets, targets, weights

# ================================================================
# Print GPU info
# ================================================================

def print_gpu_info():
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    mem = dev.mem_info
    print("=" * 50)
    print(f"GPU: {props['name'].decode()}")
    print(f"  SMs:            {props['multiProcessorCount']}")
    print(f"  Memory:         {props['totalGlobalMem'] // (1024**2)} MB")
    bw = 2.0 * props['memoryClockRate'] * (props['memoryBusWidth'] / 8) / 1e6
    print(f"  Bandwidth:      {bw:.0f} GB/s")
    print(f"  L2 Cache:       {props['l2CacheSize'] // 1024} KB")
    print(f"  Compute cap:    {props['major']}.{props['minor']}")
    print("=" * 50)
    print()

# ================================================================
# Run simulation
# ================================================================

def run_simulation(n_neurons, n_synapses, offsets, targets, weights,
                   num_timesteps=10000, warmup_steps=500, seed=42, verbose=False):

    kernels = compile_kernels()
    print("CUDA kernels compiled successfully.\n")

    spike_words = (n_neurons + 31) // 32
    BLOCK = 256
    PROP_BLOCK = 128
    MAX_PROP_BLOCKS = 2048

    neuron_blocks  = (n_neurons + BLOCK - 1) // BLOCK
    compact_blocks = (spike_words + BLOCK - 1) // BLOCK

    # Upload to GPU
    print("Uploading to GPU...")
    t0 = time.perf_counter()
    d_offsets = cp.asarray(offsets)
    d_targets = cp.asarray(targets)
    d_weights = cp.asarray(weights)

    # Initialize voltages randomly near threshold to kickstart activity
    rng_cp = cp.random.default_rng(seed)
    d_voltage    = rng_cp.uniform(0.0, 0.9, n_neurons).astype(cp.float32)
    d_current    = cp.zeros(n_neurons, dtype=cp.float32)
    d_spike_bits = cp.zeros(spike_words, dtype=cp.uint32)
    d_spike_idx  = cp.zeros(n_neurons, dtype=cp.uint32)
    d_num_spikes = cp.zeros(1, dtype=cp.uint32)
    t1 = time.perf_counter()

    gpu_mb = (offsets.nbytes + targets.nbytes + weights.nbytes +
              n_neurons * 4 * 3 + spike_words * 4 + 4) / 1e6
    print(f"  Upload time: {t1-t0:.2f}s")
    print(f"  GPU memory:  {gpu_mb:.1f} MB\n")

    # LIF parameters
    tau_decay   = np.float32(0.9)
    v_threshold = np.float32(1.0)
    v_reset     = np.float32(0.0)
    noise_amp   = np.float32(0.4)

    # Timing accumulators
    total_noise_us = 0.0
    total_update_us = 0.0
    total_compact_us = 0.0
    total_prop_us = 0.0
    total_spikes = 0

    total_steps = warmup_steps + num_timesteps

    print(f"Running: {warmup_steps} warmup + {num_timesteps} benchmark timesteps")
    print(f"{'Step':<10} {'Noise':>8} {'Update':>8} {'Compact':>8} {'Prop':>8} {'Total':>8} {'Spikes':>8} {'Rate':>7}")
    print("-" * 80)

    ev = [cp.cuda.Event() for _ in range(5)]  # start, noise, update, compact, prop

    for step in range(total_steps):
        is_bench = step >= warmup_steps

        d_num_spikes.fill(0)

        # Phase 1: Noise
        ev[0].record()
        kernels["noise"]((neuron_blocks,), (BLOCK,),
            (d_current, np.uint32(seed), np.uint32(step), noise_amp, np.int32(n_neurons)))
        ev[1].record()

        # Phase 2: Neuron update + spike detect
        kernels["update"]((neuron_blocks,), (BLOCK,),
            (d_voltage, d_current, d_spike_bits,
             np.int32(n_neurons), np.int32(spike_words),
             tau_decay, v_threshold, v_reset))
        ev[2].record()

        # Phase 3: Compact spikes
        kernels["compact"]((compact_blocks,), (BLOCK,),
            (d_spike_bits, d_spike_idx, d_num_spikes,
             np.int32(spike_words), np.int32(n_neurons)))
        ev[3].record()

        # Sync to read spike count
        cp.cuda.Stream.null.synchronize()
        num_spikes = int(d_num_spikes[0])

        # Phase 4: Propagate
        if num_spikes > 0:
            warps_per_block = PROP_BLOCK // 32
            prop_blocks = min((num_spikes + warps_per_block - 1) // warps_per_block,
                              MAX_PROP_BLOCKS)
            kernels["propagate"]((prop_blocks,), (PROP_BLOCK,),
                (d_spike_idx, np.uint32(num_spikes),
                 d_offsets, d_targets, d_weights, d_current))

        ev[4].record()
        cp.cuda.Stream.null.synchronize()

        # Timings (ms -> us)
        t_noise   = cp.cuda.get_elapsed_time(ev[0], ev[1]) * 1000
        t_update  = cp.cuda.get_elapsed_time(ev[1], ev[2]) * 1000
        t_compact = cp.cuda.get_elapsed_time(ev[2], ev[3]) * 1000
        t_prop    = cp.cuda.get_elapsed_time(ev[3], ev[4]) * 1000
        t_total   = t_noise + t_update + t_compact + t_prop

        if is_bench:
            total_noise_us   += t_noise
            total_update_us  += t_update
            total_compact_us += t_compact
            total_prop_us    += t_prop
            total_spikes     += num_spikes

        rate = 100.0 * num_spikes / n_neurons

        if verbose or \
           (step < warmup_steps and step % 500 == 0) or \
           (is_bench and (step - warmup_steps) % 1000 == 0):
            phase = "BENCH" if is_bench else "WARM"
            print(f"{phase} {step:<5} {t_noise:7.1f}  {t_update:7.1f}  {t_compact:7.1f}  "
                  f"{t_prop:7.1f}  {t_total:7.1f}  {num_spikes:7d}  {rate:5.2f}%")

    # Results
    avg_noise   = total_noise_us / num_timesteps
    avg_update  = total_update_us / num_timesteps
    avg_compact = total_compact_us / num_timesteps
    avg_prop    = total_prop_us / num_timesteps
    avg_total   = avg_noise + avg_update + avg_compact + avg_prop
    avg_spikes  = total_spikes / num_timesteps

    print()
    print("=" * 60)
    print(f"  BENCHMARK RESULTS (averaged over {num_timesteps} timesteps)")
    print("=" * 60)
    print(f"  Noise injection:    {avg_noise:8.1f} us")
    print(f"  Neuron update:      {avg_update:8.1f} us")
    print(f"  Spike compaction:   {avg_compact:8.1f} us")
    print(f"  Spike propagation:  {avg_prop:8.1f} us")
    print(f"  {'─'*40}")
    print(f"  Total per timestep: {avg_total:8.1f} us")
    print()
    print(f"  Avg spikes/step:    {avg_spikes:.0f} ({100*avg_spikes/n_neurons:.2f}% firing rate)")
    print()

    bio_per_sec = 1e6 / avg_total  # ms of bio time per wall second
    speedup = bio_per_sec / 1000.0

    print(f"  Biological time per wall-second: {bio_per_sec:.0f} ms")
    print(f"  Speed vs real-time:              {speedup:.1f}x")
    print()

    if speedup >= 1.0:
        print(f"  >>> FASTER THAN REAL-TIME <<<")
    else:
        print(f"  {1.0/speedup:.1f}x slower than real-time")

    vs_brian2 = 300.0 * speedup
    print(f"  Speed vs Brian2 (est):           {vs_brian2:.0f}x faster")
    print("=" * 60)

    # Bottleneck
    print()
    print("BOTTLENECK ANALYSIS:")
    for name, val in [("Noise injection", avg_noise), ("Neuron update", avg_update),
                       ("Spike compaction", avg_compact), ("Spike propagation", avg_prop)]:
        print(f"  {name:<20} {100*val/avg_total:5.1f}%  ({val:.1f} us)")

    avg_degree = n_synapses / n_neurons
    active_syn = avg_spikes * avg_degree
    bytes_step = active_syn * 8  # 4 bytes target + 4 bytes weight (float32)
    bw_used = bytes_step / (avg_prop * 1e-6) / 1e9 if avg_prop > 0 else 0

    print()
    print(f"  Avg degree:                 {avg_degree:.0f} synapses/neuron")
    print(f"  Est. active synapses/step:  {active_syn:.0f}")
    print(f"  Est. propagation bandwidth: {bw_used:.1f} GB/s")
    print()

# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyWire Connectome GPU Simulator")
    parser.add_argument("--data", help="Binary connectome file (from download_connectome.py)")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  FlyWire Connectome GPU Simulator")
    print("=" * 60)
    print()

    print_gpu_info()

    if args.data:
        print(f"DATA SOURCE: Real FlyWire connectome (v783)\n")
        n_neurons, n_synapses, offsets, targets, weights = load_connectome_binary(args.data)
    else:
        print("DATA SOURCE: Synthetic (use --data flywire_v783.bin for real data)\n")
        n_neurons, n_synapses, offsets, targets, weights = generate_synthetic(seed=args.seed)

    print(f"Connectome: {n_neurons} neurons, {n_synapses} synapses\n")

    run_simulation(n_neurons, n_synapses, offsets, targets, weights,
                   num_timesteps=args.timesteps, warmup_steps=args.warmup,
                   seed=args.seed, verbose=args.verbose)
