"""
FastFly on Modal — run the full brain simulator on a cloud GPU from your Mac.

Setup (on your Mac):
    pip install modal
    python3 -m modal setup

Run:
    modal serve modal_app.py      # dev mode — live URL, hot reloads
    modal deploy modal_app.py     # production — persistent URL
"""

import modal
from pathlib import Path

LOCAL_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Container image
#    - CUDA + Python packages + connectome data are baked in (cached)
#    - App source code is added via add_local_dir (fresh on each deploy)
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "cupy-cuda12x>=13.6.0",
        "fastapi>=0.129.0",
        "numpy>=1.24.0,<2.0.0",
        "uvicorn[standard]>=0.32.0",
        "pandas>=2.0.0",
        "pyarrow>=10.0.0",
        "requests",
    )
    .apt_install("git")
    .run_commands("mkdir -p /data")
    .run_commands(
        "git clone https://github.com/AqeelAqeel/FastFly.git /data-build",
        "cd /data-build && git checkout feat-glb-flybody-aqeel",
        "cd /data-build && python download_connectome.py",
        "cd /data-build && python download_metadata.py",
        "cp /data-build/flywire_v783.bin /data/flywire_v783.bin",
        "cp /data-build/neuron_annotations.npz /data/neuron_annotations.npz",
        "cp -r /data-build/static/neurons.json /data/neurons.json || true",
        "rm -rf /data-build",
    )
)

image = base_image.add_local_dir(
    LOCAL_DIR,
    remote_path="/app",
    copy=True,
    ignore=["*.venv*", "__pycache__", ".git", "*.bin", "*.npz",
            "nourse_model", "*.png", "*.jpg", "*.svg.png", "uv.lock"],
)

app = modal.App("fastfly", image=image)

# ---------------------------------------------------------------------------
# 2. Serve the FastAPI app on a T4 GPU
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    import sys, os

    for f in ["flywire_v783.bin", "neuron_annotations.npz"]:
        src, dst = f"/data/{f}", f"/app/{f}"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    sys.argv = ["app_server", "--data", "/app/flywire_v783.bin",
                "--host", "0.0.0.0", "--port", "8000"]
    sys.path.insert(0, "/app")
    from app_server import app as fastapi_app
    return fastapi_app


# ---------------------------------------------------------------------------
# 3. Quick CLI test
# ---------------------------------------------------------------------------
@app.function(gpu="T4", timeout=600)
def run_cli(timesteps: int = 1000):
    import subprocess, os
    for f in ["flywire_v783.bin", "neuron_annotations.npz"]:
        src, dst = f"/data/{f}", f"/app/{f}"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    result = subprocess.run(
        ["python", "/app/flywire_sim.py",
         "--data", "/app/flywire_v783.bin",
         "--timesteps", str(timesteps)],
        capture_output=True, text=True, cwd="/app"
    )
    return result.stdout + result.stderr


@app.local_entrypoint()
def main():
    print("Running FastFly on Modal GPU...")
    output = run_cli.remote(timesteps=1000)
    print(output)
