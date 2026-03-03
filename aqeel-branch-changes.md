# Aqeel Branch Changes & Setup Guide

Branch: `feat-glb-flybody-aqeel`
Fork: [github.com/AqeelAqeel/FastFly](https://github.com/AqeelAqeel/FastFly)
Based on: `feat-glb-flybody` from [eonfathom/FastFly](https://github.com/eonfathom/FastFly)

---

## What Is This Project?

FastFly is a GPU-accelerated simulator of a real fruit fly brain. Scientists mapped every single neuron (139,255) and synapse (54.5 million connections) in an actual *Drosophila melanogaster* brain using electron microscopy. This project takes that wiring diagram (called a **connectome**) and simulates it in real time on an NVIDIA GPU.

You can poke the virtual fly's body parts (antennae, eyes, legs, proboscis) and watch electrical signals ripple through its brain in a 3D visualizer.

**This is NOT an AI/ML model.** There are no "trained weights." The synaptic weights come directly from biology — scientists physically measured them from a real fly brain.

---

## Key Concepts

| Term | What it means |
|---|---|
| **Connectome** | A complete map of every neuron and every connection in a brain. Like a wiring diagram. |
| **Neuron** | A brain cell that receives signals, adds them up, and fires a spike if the total crosses a threshold. |
| **Synapse** | A connection between two neurons. Has a **weight** (strength). Positive = excitatory (pushes toward firing), Negative = inhibitory (pushes away). |
| **LIF (Leaky Integrate-and-Fire)** | The math model for each neuron: incoming signals add to voltage, voltage leaks over time, fires when it hits threshold (1.0), then resets. |
| **CSR (Compressed Sparse Row)** | Data structure for storing the connectome efficiently. Only stores actual connections, not the empty space. |
| **CUDA** | NVIDIA's framework for running code on GPU processors. Thousands of tiny cores update all neurons in parallel. |
| **CuPy** | Python library that lets you run CUDA GPU code without writing C++. Used by the Python simulator. |
| **Spike propagation (push model)** | When a neuron fires, it pushes its signal to all connected neurons. Only processes connections from neurons that actually fired (~1% per step). |
| **FP16 / INT8** | Smaller number formats that use less memory. FP16 = 16-bit float, INT8 = 8-bit integer. Less memory = faster simulation. |
| **Cold start** | When a cloud GPU container has been idle and needs ~10-30 seconds to boot up on the first request. |

---

## What Changed on This Branch

### Commits by Aqeel (on top of feat-glb-flybody)

1. **`192ca83` — feat: add Modal deployment + fix cloud WebSocket/API URLs**
   - Created `modal_app.py` for one-command cloud GPU deployment
   - Changed `static/index.html` to auto-detect API/WebSocket URLs (works on both localhost and cloud deployments)
   - Added a cold-start loading overlay so users see "Waking up GPU container..." instead of a blank error page

2. **`442071f` — fix: Modal deploy config + cold-start overlay**
   - Refined Modal image build layers for better caching

3. **`091fd53` — fix: simplify Modal deploy — code from git, force rebuild**
   - Simplified deployment: code comes from git clone inside the container image
   - Added `force_build=True` to ensure fresh code on each deploy

### Files added/modified

| File | Change |
|---|---|
| `modal_app.py` | **NEW** — Modal deployment script |
| `static/index.html` | **MODIFIED** — Auto-detect cloud vs localhost URLs, added cold-start loading screen |

---

## Git Setup

### Remotes

| Remote | URL | Purpose |
|---|---|---|
| `origin` | github.com/AqeelAqeel/FastFly | Your fork — push here |
| `upstream` | github.com/eonfathom/FastFly | Original repo — pull updates from here |

### Branches on your fork

| Branch | What it is |
|---|---|
| `main` | Base code from original repo |
| `feat-glb-flybody` | Advanced branch (3D fly model, experiments, AWS deploy) |
| `feat-io-experiments` | I/O experiments branch |
| `feat-glb-flybody-aqeel` | **Your working branch** |

### Common git operations

```bash
# Push your changes
git add .
git commit -m "your message"
git push

# Pull updates from the original repo
git fetch upstream
git merge upstream/feat-glb-flybody
```

---

## How to Run (Cloud GPU via Modal)

You don't need a GPU on your Mac. Everything runs on Modal's cloud GPUs.

### One-Time Setup

```bash
# 1. Create a Python virtual environment
cd /path/to/FastFly
uv venv .venv        # or: python3 -m venv .venv
source .venv/bin/activate

# 2. Install the Modal client
pip install modal

# 3. Authenticate (opens browser)
python3 -m modal setup
```

### Deploy the Simulator

```bash
source .venv/bin/activate
modal deploy modal_app.py
```

This gives you a permanent URL: **https://aqeelaqeel--fastfly-serve.modal.run**

The URL stays up even when your Mac is off. The GPU container boots on demand when someone visits (~10-30 second cold start), then stays warm for 5 minutes between requests.

### Quick CLI Test (Optional)

```bash
source .venv/bin/activate
modal run modal_app.py
```

Runs 1000 simulation timesteps on a cloud GPU and prints performance stats to your terminal.

### Dev Mode (Hot Reload)

```bash
source .venv/bin/activate
modal serve modal_app.py
```

Gives you a temporary `-dev` URL. Auto-reloads when you change code. Stops when you close the terminal.

---

## How to Update After Code Changes

```bash
# 1. Make your changes locally
# 2. Commit and push to GitHub
git add .
git commit -m "description of changes"
git push

# 3. Redeploy to Modal (pulls latest code from GitHub)
source .venv/bin/activate
modal deploy modal_app.py
```

The `force_build=True` flag in `modal_app.py` ensures Modal always pulls the latest code from your GitHub repo. Remove this flag once you're done actively iterating — it will make deploys faster by caching the data download layer.

---

## How the Modal Deployment Works

`modal_app.py` tells Modal to:

1. **Build a container image** (cached after first build):
   - Start from NVIDIA CUDA 12.2 base image (Ubuntu 22.04 + GPU drivers)
   - Install Python packages (CuPy, FastAPI, NumPy, etc.)
   - Clone your GitHub repo
   - Run `download_connectome.py` → downloads the fly brain wiring data (~120 MB binary)
   - Run `download_metadata.py` → downloads neuron annotations (cell types, 3D positions)

2. **Serve the FastAPI web app** on a T4 GPU:
   - The `serve()` function returns the FastAPI ASGI app
   - Modal handles HTTPS, WebSocket upgrades, and auto-scaling
   - Container scales to zero after 5 minutes idle (saves money)
   - Boots back up in ~10-30 seconds on next request

### Cost

| Resource | Cost |
|---|---|
| T4 GPU on Modal | ~$0.59/hour |
| Idle (scaled to zero) | $0.00 |
| Container boot (cold start) | Free (billed only while running) |

Your $30 in Modal credits = ~50 hours of GPU time.

---

## Where the Brain Data Comes From

There are NO pre-trained weights. The data comes from real neuroscience:

### Synaptic connectivity (the "weights")
- **Source**: [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Paper**: Shiu et al. 2024, *Nature*
- **What it is**: An edge list of every synapse in the fly brain. "Neuron A connects to Neuron B with strength X."
- **Downloaded by**: `download_connectome.py` → produces `flywire_v783.bin`
- **Size**: 138,639 neurons, 15M synapses, ~120 MB binary file

### Neuron annotations (cell types, positions)
- **Source**: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Paper**: Schlegel et al. 2024, *Nature*
- **What it is**: Labels for each neuron — what type it is (sensory, motor, visual, etc.), where it is in the brain, which body part it connects to
- **Downloaded by**: `download_metadata.py` → produces `neuron_annotations.npz`

### The underlying brain scan
- **Source**: [FlyWire project](https://flywire.ai/) — crowd-sourced neuron tracing on the FAFB electron microscopy volume
- You don't need to download this — the processed data above is already extracted from it

---

## Project File Reference

| File | Description |
|---|---|
| `modal_app.py` | Modal cloud deployment script (your addition) |
| `app_server.py` | FastAPI web server with WebSocket live visualizer |
| `app.py` | Simple entry point that runs app_server with uvicorn |
| `sim_engine.py` | Simulation engine wrapping CuPy CUDA kernels |
| `flywire_sim.py` | Python/CuPy simulator (runtime-compiled CUDA kernels) |
| `flywire_sim.cu` | Standalone C++ CUDA simulator |
| `torch_engine.py` | Alternative PyTorch-based simulator (slower, more biologically accurate) |
| `download_connectome.py` | Downloads FlyWire connectome data → `flywire_v783.bin` |
| `download_metadata.py` | Downloads neuron annotations → `neuron_annotations.npz` |
| `static/index.html` | Main web UI — 3D brain, fly body, charts, motor output |
| `static/io_analysis.html` | I/O analysis page (CuPy engine) |
| `static/io_analysis_torch.html` | I/O analysis page (PyTorch engine) |
| `run_experiments.py` | Headless I/O experiment runner |
| `characterize_GRN_DNa02_pathway.py` | Specific neural pathway analysis script |
| `convert_flybody.py` | Converts 3D fly body model to web format |
| `Dockerfile` | Docker container for EC2 deployment |
| `docker-compose.yml` | Docker Compose config for EC2 |
| `ec2-setup.sh` | EC2 instance setup script |
| `DEPLOYMENT.md` | Full EC2 + Vercel deployment guide |
| `DEPLOY-QUICKREF.md` | Quick deployment reference |

---

## GPU Memory Requirements

The entire fly brain fits in under 1 GB of GPU memory:

| Data | Size |
|---|---|
| Synapse targets (54.5M x 4 bytes) | 218 MB |
| Synapse weights INT8 (54.5M x 1 byte) | 55 MB |
| CSR offsets + neuron state | ~5 MB |
| CuPy/CUDA runtime overhead | ~300-500 MB |
| **Total** | **~600-800 MB** |

Any NVIDIA GPU with 4GB+ VRAM works. The T4 on Modal has 16 GB — massive overkill for this.

---

## Troubleshooting

### "modal-http: app for invoked web endpoint is stopped"
The GPU container scaled to zero after 5 minutes idle. Just refresh the page — it takes ~10-30 seconds to boot. You'll see the "Waking up GPU container..." loading screen.

### WebSocket won't connect
The frontend auto-detects the URL based on `location.host`. If you see WebSocket errors in the browser console, check that your Modal app is deployed and the URL matches.

### Deploy takes a long time
First deploy: ~2-3 minutes (building image, downloading connectome). Subsequent deploys: ~5-10 seconds (cached layers). If `force_build=True` is set, each deploy re-clones from GitHub which adds ~2 minutes.

### "Permission denied" on git push
Make sure you're pushing to `origin` (your fork), not `upstream` (the original repo):
```bash
git remote -v   # check remotes
git push origin feat-glb-flybody-aqeel
```

### Modal credits running low
Check your balance at [modal.com/settings](https://modal.com/settings). The T4 costs ~$0.59/hr but only bills while a container is running. Idle = free.
