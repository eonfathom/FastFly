# Claude Working Notes

## 2026-02-10: Performance Optimization Pass

### Starting Point
- RTX 5080 (Blackwell, SM 12.0, 84 SMs, 16GB VRAM, ~960 GB/s bandwidth)
- ~3-4x real-time on real FlyWire connectome (138,639 neurons, 15.1M synapses)
- Spike propagation kernel was ~70-80% of step time, memory-bound
- Weights were FP32 (4 bytes each) — the dominant bandwidth cost

### What I Changed

**1. INT8 Weight Quantization (biggest win)**
- FP32 weights (4 bytes) → INT8 (1 byte) with per-neuron float32 scale factors
- Dequantization in kernel: `float w = (float)weights[syn] * scale[neuron]`
- Scale factor loaded once per spiking neuron, amortized across its entire fan-out
- Quantization uses `np.maximum.reduceat` — vectorized, no Python loop over 139K neurons
- Saves ~163 MB VRAM, cuts weight bandwidth 4x
- Per-neuron scales (not global) because audit showed global scale gives only ~13 usable quantization levels for mean-weight neurons

**2. Fused Noise + Update Kernel**
- Merged `kernel_inject_noise` + `kernel_update_neurons` → `kernel_update_with_noise`
- Both had identical grid config `(neuron_blocks,) x (256,)` and same thread mapping
- Key benefit: `current[i]` stays in register instead of write-to-global then read-back
- Saves ~543 KB global memory traffic per step + eliminates one kernel launch

**3. Batch Size 50 → 200**
- Free speedup from fewer GPU-CPU synchronization points
- sim_engine.py already has single-sync-at-batch-end design, so larger batches = less Python overhead

**4. Throttled active_indices Transfer**
- GPU→CPU transfer of spiking neuron indices now happens every 3rd batch instead of every batch
- Brain 3D visualizer only needs ~20fps; at batch_size=200 this is still plenty
- Saves GPU readback + JSON serialization overhead

### What I Considered But Didn't Do

**Folding `d_num_spikes.fill(0)` into compact kernel** — I initially implemented this (thread 0 of block 0 writes `*num_spikes = 0` with `__threadfence`), but realized there's a cross-block race: block N could `atomicAdd` before block 0's write propagates. Would need cooperative groups grid sync, which adds complexity for a tiny gain. Reverted.

**FP16 weights** — Skipped as intermediate step, went straight to INT8 since the per-neuron scale infrastructure is the same either way and INT8 gives 2x more bandwidth savings.

**Shared-memory accumulation buffers** — Audit showed warp-internal target collisions are ~0.4% at 109 mean out-degree / 139K neurons. Expected speedup ~1.005x. Not worth the complexity.

**CUDA Graphs** — `step` param in noise kernel would be frozen during replay (breaking randomness), and conditional stimulus is incompatible with graph capture. Would need device-side step counter. Too complex for ~5-10% gain.

### Results

| Metric | Before | After |
|---|---|---|
| Speed vs real-time | ~3-4x | **7.8x** |
| Total per timestep | ~300 us | **128.1 us** |
| Propagation kernel | ~200+ us | 95.1 us |
| Weight memory | ~60 MB | ~15 MB + 0.5 MB scales |
| Kernels per step | 5 | 4 |

Propagation is still the bottleneck at 74.2% — it's fundamentally memory-bound. Further gains would need architectural changes (pull model at higher firing rates, or connectivity pruning).

### Remaining Optimization Ideas (not yet pursued)
- Multi-stream pipelining (overlap compact + propagate across substeps)
- Connectivity pruning (remove weakest synapses to reduce total work)
- Warp-cooperative target sorting to batch atomicAdds to same cache lines
- Pull model comparison if firing rates go above ~5%
