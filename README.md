# FastFly

GPU-accelerated simulator for the complete *Drosophila melanogaster* (fruit fly) brain connectome — 139,255 neurons and 54.5 million synapses — targeting real-time or faster performance on a single consumer NVIDIA GPU.

![Fruit fly brain](fruitfly.jpg)

## How it works

- **Neuron model:** Leaky Integrate-and-Fire (LIF), validated by [arXiv:2404.17128](https://arxiv.org/abs/2404.17128)
- **Connectivity:** CSR sparse format with FP16 synaptic weights
- **Spike propagation:** Push model — only processes synapses of neurons that actually fired
- **Spike detection:** Warp ballot intrinsics for bit-packed spike flags
- **Load balancing:** Warp-per-spike with grid-stride loop

## Data sources

This simulator uses real connectome data from the [FlyWire](https://flywire.ai/) project — a collaborative effort to map every neuron and synapse in an adult *Drosophila melanogaster* brain from electron microscopy imagery.

### Synaptic connectivity

- **Data:** `Connectivity_783.parquet` and `Completeness_783.csv` (FlyWire materialization v783)
- **Repository:** [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Paper:** Shiu PK, Sterne GR, Spiller N, et al. "A Drosophila computational brain model reveals sensorimotor processing." *Nature* 634, 210–219 (2024). [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)
- **Contents:** 139,255 neurons, 54.5M synaptic connections with signed excitatory/inhibitory weights in CSR sparse format

### Neuron annotations (cell types, positions, neurotransmitters)

- **Data:** `Supplemental_file1_neuron_annotations.tsv`
- **Repository:** [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Paper:** Schlegel P, Yin Y, Bates AS, et al. "Whole-brain annotation and multi-connectome cell typing of Drosophila." *Nature* 634, 139–152 (2024). [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)
- **Contents:** Cell type classifications (super_class, cell_class, cell_type), neurotransmitter identity, laterality, nerve assignments, and 3D soma positions in FAFB voxel coordinates (4×4×40 nm resolution)

### Underlying electron microscopy volume

- **Dataset:** FAFB (Full Adult Fly Brain)
- **Paper:** Zheng Z, Lauritzen JS, Perlman E, et al. "A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster." *Cell* 174(3), 730–743 (2018). [doi:10.1016/j.cell.2018.06.019](https://doi.org/10.1016/j.cell.2018.06.019)

### FlyWire connectome

- **Platform:** [flywire.ai](https://flywire.ai/) · [Codex browser](https://codex.flywire.ai/)
- **Paper:** Dorkenwald S, Matsliah A, Sterling AR, et al. "Neuronal wiring diagram of an adult brain." *Nature* 634, 124–138 (2024). [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)

### Neuron model validation

- Zhang X, Yang P, Feng J, et al. "Network Structure Governs Drosophila Brain Functionality." [arXiv:2404.17128](https://arxiv.org/abs/2404.17128) (2024). Demonstrates that network structure dominates over neuron model choice, validating the use of LIF for whole-brain simulation.

## Requirements

- **CUDA path (C++):** NVIDIA CUDA Toolkit 12.x
- **Python path:** Python 3.10+, [CuPy](https://cupy.dev/) (`pip install cupy-cuda12x`)
- **Web visualizer:** FastAPI, uvicorn (`pip install fastapi uvicorn[standard]`)

## Quick start

### 1. Download the connectome

```bash
pip install pandas pyarrow numpy requests
python download_connectome.py
```

This downloads the FlyWire v783 data and produces `flywire_v783.bin`.

### 2a. Run the CUDA simulator (C++)

```bash
build.bat                               # compile (requires nvcc)
flywire_sim.exe --data flywire_v783.bin  # run with real connectome
flywire_sim.exe                          # or run with synthetic data
```

### 2b. Run the Python/CuPy simulator

```bash
pip install cupy-cuda12x
python flywire_sim.py --data flywire_v783.bin
python flywire_sim.py                    # synthetic data fallback
```

### 3. Web visualizer

```bash
pip install fastapi uvicorn[standard]
python app_server.py --data flywire_v783.bin
# Open http://127.0.0.1:8000
```

## Project structure

| File | Description |
|---|---|
| `flywire_sim.cu` | Standalone CUDA C++ simulator |
| `flywire_sim.py` | Python/CuPy simulator (runtime-compiled CUDA kernels, no Visual Studio needed) |
| `sim_engine.py` | Simulation engine used by the web server |
| `app_server.py` | FastAPI web server with WebSocket-based live visualizer |
| `download_connectome.py` | Downloads FlyWire v783 data and converts to binary format |
| `download_metadata.py` | Downloads neuron annotation metadata |
| `build.bat` | Build script for the C++ simulator (targets RTX 3080 Ti / SM 8.6) |

## License

MIT
