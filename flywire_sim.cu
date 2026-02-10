// ================================================================
// FlyWire Connectome GPU Simulator
// ================================================================
//
// Simulates the complete Drosophila (fruit fly) brain connectome
// as fast as possible on a single consumer GPU.
//
// Supports two data modes:
//   --data flywire_v783.bin   Load real FlyWire connectome (from download_connectome.py)
//   (no --data flag)          Generate synthetic connectome matching FlyWire statistics
//
// Architecture:
//   - Push-model spike propagation (only touches active synapses)
//   - FP16 synaptic weights (doubles effective memory bandwidth)
//   - Bit-packed spike detection via warp ballot intrinsics
//   - Stream compaction for spiking neuron indices
//   - Per-warp spike propagation with grid-stride load balancing
//
// Build: nvcc -O3 -arch=sm_86 --use_fast_math -o flywire_sim.exe flywire_sim.cu
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>

// ================================================================
// Configuration
// ================================================================

// Neuron model parameters (Leaky Integrate-and-Fire)
#define V_REST        0.0f
#define V_THRESHOLD   1.0f
#define V_RESET       0.0f
#define TAU_DECAY     0.9f    // exp(-dt/tau), ~10ms time constant at 1ms step

// Simulation defaults
#define DEFAULT_TIMESTEPS     10000
#define DEFAULT_WARMUP_STEPS  500
#define NOISE_AMPLITUDE       0.15f

// Kernel launch config
#define BLOCK_SIZE       256
#define PROP_BLOCK_SIZE  128
#define MAX_PROP_BLOCKS  2048

// Binary file format
#define FLYWIRE_MAGIC   0x464C5957  // "FLYW"
#define FLYWIRE_VERSION 1

// ================================================================
// Error checking
// ================================================================

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// ================================================================
// Data structures
// ================================================================

struct SimConfig {
    int      num_neurons;
    int      num_synapses;
    int      spike_words;   // (num_neurons + 31) / 32
    int      num_timesteps;
    int      warmup_steps;
    bool     verbose;
    unsigned seed;
    char     data_file[512];
    bool     use_real_data;
};

struct DeviceConnectome {
    uint32_t* offsets;     // [num_neurons + 1]
    uint32_t* targets;     // [num_synapses]
    half*     weights;     // [num_synapses]
};

struct DeviceNeurons {
    float*    voltage;
    float*    current;
    uint32_t* spike_bits;
    uint32_t* spike_idx;
    uint32_t* num_spikes;  // managed memory
};

struct TimingResult {
    float neuron_update_us;
    float compaction_us;
    float propagation_us;
    float noise_us;
    float total_us;
    float spikes_per_step;
};

// ================================================================
// KERNEL: Random noise injection
// ================================================================

__global__ void kernel_inject_noise(
    float* __restrict__ current,
    const uint32_t       seed,
    const uint32_t       step,
    const float          amplitude,
    const int            N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t h = i ^ (step * 2654435761u) ^ (seed * 1664525u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;

    float noise = amplitude * (((float)(h & 0xFFFF) / 32768.0f) - 1.0f);
    current[i] += noise;
}

// ================================================================
// KERNEL: Neuron update + spike detection (fused)
// ================================================================

__global__ void kernel_update_neurons(
    float*    __restrict__ voltage,
    float*    __restrict__ current,
    uint32_t* __restrict__ spike_bits,
    const int              N,
    const int              spike_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool spiked = false;
    if (i < N) {
        float v = voltage[i] * TAU_DECAY + current[i];
        spiked = (v >= V_THRESHOLD);
        voltage[i] = spiked ? V_RESET : v;
        current[i] = 0.0f;
    }

    uint32_t ballot = __ballot_sync(0xFFFFFFFF, spiked);
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        int word = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        if (word < spike_words) {
            spike_bits[word] = ballot;
        }
    }
}

// ================================================================
// KERNEL: Stream compaction - extract spiking neuron indices
// ================================================================

__global__ void kernel_compact_spikes(
    const uint32_t* __restrict__ spike_bits,
    uint32_t*       __restrict__ spike_idx,
    uint32_t*       __restrict__ num_spikes,
    const int                    num_words,
    const int                    N
) {
    __shared__ uint32_t s_offsets[BLOCK_SIZE];
    __shared__ uint32_t block_base;

    int tid = threadIdx.x;
    int word_idx = blockIdx.x * blockDim.x + tid;

    uint32_t bits = 0;
    int count = 0;
    if (word_idx < num_words) {
        bits = spike_bits[word_idx];
        count = __popc(bits);
    }
    s_offsets[tid] = count;
    __syncthreads();

    // Inclusive prefix sum in shared memory
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        uint32_t val = 0;
        if (tid >= stride) val = s_offsets[tid - stride];
        __syncthreads();
        s_offsets[tid] += val;
        __syncthreads();
    }

    uint32_t block_total = s_offsets[BLOCK_SIZE - 1];
    uint32_t my_offset = (tid > 0) ? s_offsets[tid - 1] : 0;

    if (tid == 0 && block_total > 0) {
        block_base = atomicAdd(num_spikes, block_total);
    }
    __syncthreads();

    if (word_idx < num_words && bits != 0) {
        uint32_t base_neuron = word_idx * 32;
        uint32_t write_pos = block_base + my_offset;
        while (bits) {
            int b = __ffs(bits) - 1;
            uint32_t neuron = base_neuron + b;
            if (neuron < (uint32_t)N) {
                spike_idx[write_pos++] = neuron;
            }
            bits &= bits - 1;
        }
    }
}

// ================================================================
// KERNEL: Spike propagation (push model, warp-per-spike)
// ================================================================

__global__ void kernel_propagate_spikes(
    const uint32_t* __restrict__ spike_idx,
    const uint32_t               num_spikes,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ targets,
    const half*     __restrict__ weights,
    float*          __restrict__ current
) {
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t num_warps = (gridDim.x * blockDim.x) >> 5;
    uint32_t lane = threadIdx.x & 31;

    for (uint32_t s = warp_id; s < num_spikes; s += num_warps) {
        uint32_t neuron = spike_idx[s];
        uint32_t syn_start = offsets[neuron];
        uint32_t syn_end   = offsets[neuron + 1];

        for (uint32_t syn = syn_start + lane; syn < syn_end; syn += 32) {
            uint32_t t = targets[syn];
            float    w = __half2float(weights[syn]);
            atomicAdd(&current[t], w);
        }
    }
}

// ================================================================
// HOST: Load real connectome from binary file
// ================================================================

bool load_connectome_binary(
    const char* filename,
    std::vector<uint32_t>& h_offsets,
    std::vector<uint32_t>& h_targets,
    std::vector<half>&     h_weights,
    int& num_neurons,
    int& num_synapses
) {
    printf("Loading connectome from '%s'...\n", filename);

    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open '%s'\n", filename);
        fprintf(stderr, "Run 'python download_connectome.py' first to download the data.\n");
        return false;
    }

    // Read header
    uint32_t magic, version;
    uint32_t n_neurons, n_synapses;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&n_neurons, 4, 1, f);
    fread(&n_synapses, 4, 1, f);

    if (magic != FLYWIRE_MAGIC) {
        fprintf(stderr, "ERROR: Invalid file magic (expected 0x%X, got 0x%X)\n",
                FLYWIRE_MAGIC, magic);
        fclose(f);
        return false;
    }
    if (version != FLYWIRE_VERSION) {
        fprintf(stderr, "ERROR: Unsupported file version %u (expected %u)\n",
                version, FLYWIRE_VERSION);
        fclose(f);
        return false;
    }

    num_neurons = (int)n_neurons;
    num_synapses = (int)n_synapses;

    printf("  Neurons:  %d\n", num_neurons);
    printf("  Synapses: %d\n", num_synapses);

    // Read CSR offsets
    h_offsets.resize(num_neurons + 1);
    size_t read = fread(h_offsets.data(), sizeof(uint32_t), num_neurons + 1, f);
    if (read != (size_t)(num_neurons + 1)) {
        fprintf(stderr, "ERROR: Failed to read offsets (got %zu, expected %d)\n",
                read, num_neurons + 1);
        fclose(f);
        return false;
    }

    // Read targets
    h_targets.resize(num_synapses);
    read = fread(h_targets.data(), sizeof(uint32_t), num_synapses, f);
    if (read != (size_t)num_synapses) {
        fprintf(stderr, "ERROR: Failed to read targets (got %zu, expected %d)\n",
                read, num_synapses);
        fclose(f);
        return false;
    }

    // Read weights (stored as float32, convert to FP16)
    std::vector<float> h_weights_f32(num_synapses);
    read = fread(h_weights_f32.data(), sizeof(float), num_synapses, f);
    if (read != (size_t)num_synapses) {
        fprintf(stderr, "ERROR: Failed to read weights (got %zu, expected %d)\n",
                read, num_synapses);
        fclose(f);
        return false;
    }
    fclose(f);

    // Convert weights to FP16
    h_weights.resize(num_synapses);
    for (int i = 0; i < num_synapses; i++) {
        h_weights[i] = __float2half(h_weights_f32[i]);
    }
    h_weights_f32.clear();
    h_weights_f32.shrink_to_fit();

    // Print stats
    uint32_t max_deg = 0, min_deg = UINT32_MAX;
    for (int i = 0; i < num_neurons; i++) {
        uint32_t deg = h_offsets[i + 1] - h_offsets[i];
        if (deg > max_deg) max_deg = deg;
        if (deg < min_deg) min_deg = deg;
    }
    printf("  Out-degree: min=%u, max=%u, mean=%.1f\n",
           min_deg, max_deg, (double)num_synapses / num_neurons);

    int n_exc = 0;
    for (int i = 0; i < std::min(num_synapses, 1000000); i++) {
        if (__half2float(h_weights[i]) > 0) n_exc++;
    }
    float exc_pct = 100.0f * n_exc / std::min(num_synapses, 1000000);
    printf("  Excitatory (sample): %.1f%%\n", exc_pct);
    printf("  File: offsets=%.1f MB, targets=%.1f MB, weights=%.1f MB\n",
           (num_neurons + 1) * 4.0 / 1e6,
           (long long)num_synapses * 4.0 / 1e6,
           (long long)num_synapses * 2.0 / 1e6);
    printf("  Loaded successfully.\n\n");
    return true;
}

// ================================================================
// HOST: Generate synthetic connectome (fallback)
// ================================================================

void generate_connectome(
    std::vector<uint32_t>& h_offsets,
    std::vector<uint32_t>& h_targets,
    std::vector<half>&     h_weights,
    int num_neurons,
    long long num_synapses,
    unsigned seed
) {
    printf("Generating synthetic connectome (%d neurons, %lld synapses)...\n",
           num_neurons, num_synapses);

    std::mt19937 rng(seed);

    double mean_degree = (double)num_synapses / num_neurons;
    double log_mean = log(mean_degree) - 0.5 * log(1.0 + 4.0);
    double log_std = sqrt(log(1.0 + 4.0));

    std::lognormal_distribution<double> degree_dist(log_mean, log_std);
    std::vector<uint32_t> out_degree(num_neurons);

    long long total = 0;
    for (int i = 0; i < num_neurons; i++) {
        int d = (int)std::max(1.0, std::min((double)(num_neurons - 1), degree_dist(rng)));
        out_degree[i] = d;
        total += d;
    }

    double scale = (double)num_synapses / total;
    total = 0;
    for (int i = 0; i < num_neurons; i++) {
        out_degree[i] = (uint32_t)std::max(1.0, round(out_degree[i] * scale));
        total += out_degree[i];
    }
    while (total != num_synapses) {
        int idx = rng() % num_neurons;
        if (total < num_synapses && out_degree[idx] < (uint32_t)(num_neurons - 1)) {
            out_degree[idx]++; total++;
        } else if (total > num_synapses && out_degree[idx] > 1) {
            out_degree[idx]--; total--;
        }
    }

    h_offsets.resize(num_neurons + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < num_neurons; i++) {
        h_offsets[i + 1] = h_offsets[i] + out_degree[i];
    }

    h_targets.resize(num_synapses);
    h_weights.resize(num_synapses);

    std::uniform_int_distribution<uint32_t> target_dist(0, num_neurons - 1);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::bernoulli_distribution excitatory_dist(0.7);

    printf("  Generating synaptic connections...\n");
    for (int i = 0; i < num_neurons; i++) {
        uint32_t start = h_offsets[i];
        uint32_t end = h_offsets[i + 1];
        bool is_excitatory = excitatory_dist(rng);

        for (uint32_t s = start; s < end; s++) {
            uint32_t t;
            do { t = target_dist(rng); } while (t == (uint32_t)i);
            h_targets[s] = t;

            float w = fabsf(weight_dist(rng)) + 0.005f;
            if (!is_excitatory) w = -w;
            h_weights[s] = __float2half(w);
        }

        // Sort targets within row for memory coalescing
        std::vector<std::pair<uint32_t, half>> pairs(end - start);
        for (uint32_t s = start; s < end; s++) {
            pairs[s - start] = {h_targets[s], h_weights[s]};
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (uint32_t s = start; s < end; s++) {
            h_targets[s] = pairs[s - start].first;
            h_weights[s] = pairs[s - start].second;
        }

        if (i % 20000 == 0 && i > 0) printf("  ... %d / %d neurons\n", i, num_neurons);
    }

    uint32_t max_deg = *std::max_element(out_degree.begin(), out_degree.end());
    uint32_t min_deg = *std::min_element(out_degree.begin(), out_degree.end());
    printf("  Degree stats: min=%u, max=%u, mean=%.1f\n", min_deg, max_deg,
           (double)num_synapses / num_neurons);
    printf("  Done.\n\n");
}

// ================================================================
// HOST: GPU memory management
// ================================================================

void upload_connectome(
    DeviceConnectome& conn,
    const std::vector<uint32_t>& h_offsets,
    const std::vector<uint32_t>& h_targets,
    const std::vector<half>&     h_weights,
    int num_neurons,
    int num_synapses
) {
    size_t off_bytes = (size_t)(num_neurons + 1) * sizeof(uint32_t);
    size_t tgt_bytes = (size_t)num_synapses * sizeof(uint32_t);
    size_t wgt_bytes = (size_t)num_synapses * sizeof(half);

    CUDA_CHECK(cudaMalloc(&conn.offsets, off_bytes));
    CUDA_CHECK(cudaMalloc(&conn.targets, tgt_bytes));
    CUDA_CHECK(cudaMalloc(&conn.weights, wgt_bytes));

    CUDA_CHECK(cudaMemcpy(conn.offsets, h_offsets.data(), off_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(conn.targets, h_targets.data(), tgt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(conn.weights, h_weights.data(), wgt_bytes, cudaMemcpyHostToDevice));

    printf("GPU memory allocated:\n");
    printf("  Offsets:  %.1f MB\n", off_bytes / 1e6);
    printf("  Targets:  %.1f MB\n", tgt_bytes / 1e6);
    printf("  Weights:  %.1f MB\n", wgt_bytes / 1e6);
    printf("  Total:    %.1f MB\n\n", (off_bytes + tgt_bytes + wgt_bytes) / 1e6);
}

void allocate_neurons(DeviceNeurons& neurons, const SimConfig& cfg) {
    CUDA_CHECK(cudaMalloc(&neurons.voltage,    cfg.num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&neurons.current,    cfg.num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&neurons.spike_bits, cfg.spike_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&neurons.spike_idx,  cfg.num_neurons * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&neurons.num_spikes, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(neurons.voltage,    0, cfg.num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(neurons.current,    0, cfg.num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(neurons.spike_bits, 0, cfg.spike_words * sizeof(uint32_t)));
}

void free_connectome(DeviceConnectome& conn) {
    cudaFree(conn.offsets);
    cudaFree(conn.targets);
    cudaFree(conn.weights);
}

void free_neurons(DeviceNeurons& neurons) {
    cudaFree(neurons.voltage);
    cudaFree(neurons.current);
    cudaFree(neurons.spike_bits);
    cudaFree(neurons.spike_idx);
    cudaFree(neurons.num_spikes);
}

// ================================================================
// HOST: Run simulation and benchmark
// ================================================================

TimingResult run_benchmark(
    DeviceConnectome& conn,
    DeviceNeurons& neurons,
    const SimConfig& cfg
) {
    int N = cfg.num_neurons;
    int neuron_blocks  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int compact_blocks = (cfg.spike_words + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t ev_start, ev_noise, ev_update, ev_compact, ev_prop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_noise));
    CUDA_CHECK(cudaEventCreate(&ev_update));
    CUDA_CHECK(cudaEventCreate(&ev_compact));
    CUDA_CHECK(cudaEventCreate(&ev_prop));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double total_noise_us = 0, total_update_us = 0;
    double total_compact_us = 0, total_prop_us = 0;
    double total_spikes = 0;

    printf("Running simulation: %d warmup + %d benchmark timesteps\n",
           cfg.warmup_steps, cfg.num_timesteps);
    printf("%-8s  %10s  %10s  %10s  %10s  %10s  %8s  %8s\n",
           "Step", "Noise(us)", "Update(us)", "Compact(us)", "Prop(us)", "Total(us)",
           "Spikes", "Rate(%)");
    printf("--------  ----------  ----------  -----------  ----------  ----------  --------  --------\n");

    int total_steps = cfg.warmup_steps + cfg.num_timesteps;

    for (int step = 0; step < total_steps; step++) {
        bool is_benchmark = (step >= cfg.warmup_steps);

        *(neurons.num_spikes) = 0;

        // Phase 1: Inject noise
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        kernel_inject_noise<<<neuron_blocks, BLOCK_SIZE, 0, stream>>>(
            neurons.current, cfg.seed, (uint32_t)step, NOISE_AMPLITUDE, N
        );
        CUDA_CHECK(cudaEventRecord(ev_noise, stream));

        // Phase 2: Neuron update + spike detection
        kernel_update_neurons<<<neuron_blocks, BLOCK_SIZE, 0, stream>>>(
            neurons.voltage, neurons.current, neurons.spike_bits, N, cfg.spike_words
        );
        CUDA_CHECK(cudaEventRecord(ev_update, stream));

        // Phase 3: Compact spike indices
        kernel_compact_spikes<<<compact_blocks, BLOCK_SIZE, 0, stream>>>(
            neurons.spike_bits, neurons.spike_idx, neurons.num_spikes,
            cfg.spike_words, N
        );
        CUDA_CHECK(cudaEventRecord(ev_compact, stream));

        // Sync to read spike count
        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t num_spikes = *(neurons.num_spikes);

        // Phase 4: Propagate spikes
        if (num_spikes > 0) {
            int warps_per_block = PROP_BLOCK_SIZE / 32;
            int prop_blocks = ((int)num_spikes + warps_per_block - 1) / warps_per_block;
            prop_blocks = min(prop_blocks, MAX_PROP_BLOCKS);

            kernel_propagate_spikes<<<prop_blocks, PROP_BLOCK_SIZE, 0, stream>>>(
                neurons.spike_idx, num_spikes,
                conn.offsets, conn.targets, conn.weights,
                neurons.current
            );
        }
        CUDA_CHECK(cudaEventRecord(ev_prop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float t_noise, t_update, t_compact, t_prop;
        CUDA_CHECK(cudaEventElapsedTime(&t_noise,   ev_start,   ev_noise));
        CUDA_CHECK(cudaEventElapsedTime(&t_update,  ev_noise,   ev_update));
        CUDA_CHECK(cudaEventElapsedTime(&t_compact, ev_update,  ev_compact));
        CUDA_CHECK(cudaEventElapsedTime(&t_prop,    ev_compact, ev_prop));

        float t_noise_us   = t_noise   * 1000.0f;
        float t_update_us  = t_update  * 1000.0f;
        float t_compact_us = t_compact * 1000.0f;
        float t_prop_us    = t_prop    * 1000.0f;
        float t_total_us   = t_noise_us + t_update_us + t_compact_us + t_prop_us;

        if (is_benchmark) {
            total_noise_us   += t_noise_us;
            total_update_us  += t_update_us;
            total_compact_us += t_compact_us;
            total_prop_us    += t_prop_us;
            total_spikes     += num_spikes;
        }

        float rate = 100.0f * num_spikes / N;

        if (cfg.verbose ||
            (step < cfg.warmup_steps && step % 500 == 0) ||
            (is_benchmark && (step - cfg.warmup_steps) % 1000 == 0)) {
            const char* phase = is_benchmark ? "BENCH" : "WARM";
            printf("%-5s%3d  %10.1f  %10.1f  %11.1f  %10.1f  %10.1f  %8u  %7.2f%%\n",
                   phase, step, t_noise_us, t_update_us, t_compact_us,
                   t_prop_us, t_total_us, num_spikes, rate);
        }
    }

    TimingResult result;
    result.neuron_update_us = (float)(total_update_us / cfg.num_timesteps);
    result.compaction_us    = (float)(total_compact_us / cfg.num_timesteps);
    result.propagation_us   = (float)(total_prop_us / cfg.num_timesteps);
    result.noise_us         = (float)(total_noise_us / cfg.num_timesteps);
    result.total_us         = result.neuron_update_us + result.compaction_us +
                              result.propagation_us + result.noise_us;
    result.spikes_per_step  = (float)(total_spikes / cfg.num_timesteps);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_noise);
    cudaEventDestroy(ev_update);
    cudaEventDestroy(ev_compact);
    cudaEventDestroy(ev_prop);
    cudaStreamDestroy(stream);

    return result;
}

// ================================================================
// HOST: Print GPU info
// ================================================================

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("========================================\n");
    printf("GPU: %s\n", prop.name);
    printf("  SMs:            %d\n", prop.multiProcessorCount);
    printf("  Clock:          %d MHz\n", prop.clockRate / 1000);
    printf("  Memory:         %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Bandwidth:      %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  L2 Cache:       %d KB\n", prop.l2CacheSize / 1024);
    printf("  Compute cap:    %d.%d\n", prop.major, prop.minor);
    printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("========================================\n\n");
}

// ================================================================
// MAIN
// ================================================================

int main(int argc, char** argv) {
    SimConfig cfg;
    cfg.num_neurons    = 139255;       // FlyWire default
    cfg.num_synapses   = 54500000;     // FlyWire default
    cfg.num_timesteps  = DEFAULT_TIMESTEPS;
    cfg.warmup_steps   = DEFAULT_WARMUP_STEPS;
    cfg.verbose        = false;
    cfg.seed           = 42;
    cfg.use_real_data  = false;
    cfg.data_file[0]   = '\0';

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            strncpy(cfg.data_file, argv[++i], sizeof(cfg.data_file) - 1);
            cfg.use_real_data = true;
        }
        else if (strcmp(argv[i], "--timesteps") == 0 && i + 1 < argc)
            cfg.num_timesteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            cfg.warmup_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            cfg.verbose = true;
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            cfg.seed = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("FlyWire Connectome GPU Simulator\n\n");
            printf("Usage: flywire_sim [options]\n\n");
            printf("  --data FILE    Load real connectome from binary file\n");
            printf("                 (generate with: python download_connectome.py)\n");
            printf("  --timesteps N  Benchmark timesteps (default: %d)\n", DEFAULT_TIMESTEPS);
            printf("  --warmup N     Warmup timesteps (default: %d)\n", DEFAULT_WARMUP_STEPS);
            printf("  --seed N       Random seed (default: 42)\n");
            printf("  --verbose      Print every timestep\n");
            return 0;
        }
    }

    printf("\n");
    printf("================================================================\n");
    printf("  FlyWire Connectome GPU Simulator\n");
    printf("================================================================\n\n");

    print_gpu_info();

    // Load or generate connectome
    std::vector<uint32_t> h_offsets;
    std::vector<uint32_t> h_targets;
    std::vector<half>     h_weights;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (cfg.use_real_data) {
        printf("DATA SOURCE: Real FlyWire connectome (v783)\n\n");
        if (!load_connectome_binary(cfg.data_file, h_offsets, h_targets, h_weights,
                                     cfg.num_neurons, cfg.num_synapses)) {
            return 1;
        }
    } else {
        printf("DATA SOURCE: Synthetic (matching FlyWire statistics)\n");
        printf("  Use --data flywire_v783.bin for real connectome data.\n\n");
        generate_connectome(h_offsets, h_targets, h_weights,
                            cfg.num_neurons, (long long)cfg.num_synapses, cfg.seed);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();
    printf("Data loading/generation: %.1f seconds\n\n", load_sec);

    // Derived config
    cfg.spike_words = (cfg.num_neurons + 31) / 32;

    printf("Connectome: %d neurons, %d synapses\n", cfg.num_neurons, cfg.num_synapses);
    printf("Spike words: %d (%d KB)\n\n", cfg.spike_words, cfg.spike_words * 4 / 1024);

    // Upload to GPU
    DeviceConnectome conn;
    upload_connectome(conn, h_offsets, h_targets, h_weights,
                      cfg.num_neurons, cfg.num_synapses);

    // Free host arrays
    h_offsets.clear(); h_offsets.shrink_to_fit();
    h_targets.clear(); h_targets.shrink_to_fit();
    h_weights.clear(); h_weights.shrink_to_fit();

    // Allocate neuron state
    DeviceNeurons neurons;
    allocate_neurons(neurons, cfg);

    printf("Neuron state: %.1f MB\n\n",
           (cfg.num_neurons * 2 * sizeof(float) +
            cfg.spike_words * sizeof(uint32_t) +
            cfg.num_neurons * sizeof(uint32_t)) / 1e6);

    // Run benchmark
    TimingResult result = run_benchmark(conn, neurons, cfg);

    // Report
    printf("\n");
    printf("================================================================\n");
    printf("  BENCHMARK RESULTS (averaged over %d timesteps)\n", cfg.num_timesteps);
    if (cfg.use_real_data)
        printf("  Data: Real FlyWire connectome v783\n");
    else
        printf("  Data: Synthetic\n");
    printf("================================================================\n");
    printf("  Noise injection:    %8.1f us\n", result.noise_us);
    printf("  Neuron update:      %8.1f us\n", result.neuron_update_us);
    printf("  Spike compaction:   %8.1f us\n", result.compaction_us);
    printf("  Spike propagation:  %8.1f us\n", result.propagation_us);
    printf("  ----------------------------------------\n");
    printf("  Total per timestep: %8.1f us\n", result.total_us);
    printf("\n");
    printf("  Avg spikes/step:    %.0f (%.2f%% firing rate)\n",
           result.spikes_per_step,
           100.0f * result.spikes_per_step / cfg.num_neurons);
    printf("\n");

    float bio_time_per_sec = 1.0e6f / result.total_us;
    float speedup = bio_time_per_sec / 1000.0f;

    printf("  Biological time per wall-second: %.0f ms\n", bio_time_per_sec);
    printf("  Speed vs real-time:              %.1fx\n", speedup);
    printf("\n");

    if (speedup >= 1.0f)
        printf("  >>> FASTER THAN REAL-TIME <<<\n");
    else {
        printf("  %.1fx slower than real-time\n", 1.0f / speedup);
        printf("  Need %.1fx more speedup to reach real-time\n", 1.0f / speedup);
    }

    float vs_brian2 = 300.0f * speedup;
    printf("  Speed vs Brian2 (est):           %.0fx faster\n", vs_brian2);
    printf("================================================================\n\n");

    // Bottleneck analysis
    printf("BOTTLENECK ANALYSIS:\n");
    float total = result.total_us;
    printf("  Noise injection:   %5.1f%%  (%.1f us)\n",
           100.0f * result.noise_us / total, result.noise_us);
    printf("  Neuron update:     %5.1f%%  (%.1f us)\n",
           100.0f * result.neuron_update_us / total, result.neuron_update_us);
    printf("  Spike compaction:  %5.1f%%  (%.1f us)\n",
           100.0f * result.compaction_us / total, result.compaction_us);
    printf("  Spike propagation: %5.1f%%  (%.1f us)\n",
           100.0f * result.propagation_us / total, result.propagation_us);
    printf("\n");

    float avg_degree = (float)cfg.num_synapses / cfg.num_neurons;
    float active_synapses = result.spikes_per_step * avg_degree;
    float bytes_per_step = active_synapses * 6.0f;
    float prop_time_sec = result.propagation_us * 1e-6f;
    float bandwidth_used = (prop_time_sec > 0) ?
                           (bytes_per_step / prop_time_sec / 1e9f) : 0;

    printf("  Avg degree:                 %.0f synapses/neuron\n", avg_degree);
    printf("  Est. active synapses/step:  %.0f\n", active_synapses);
    printf("  Est. propagation bandwidth: %.1f GB/s\n", bandwidth_used);
    printf("\n");

    free_connectome(conn);
    free_neurons(neurons);

    return 0;
}
