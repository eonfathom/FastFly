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
    .run_commands("mkdir -p /data && mkdir -p /app")
    .run_commands(
        "git clone https://github.com/AqeelAqeel/FastFly.git /data-build"
        " && cd /data-build && git checkout feat-glb-flybody-aqeel"
        " && python download_connectome.py"
        " && python download_metadata.py"
        " && cp flywire_v783.bin /data/"
        " && cp neuron_annotations.npz /data/"
        " && cp -r . /app/"
        " && rm -rf /data-build",
        force_build=True,
    )
)

image = base_image

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
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass

    sys.argv = ["app_server", "--data", "/app/flywire_v783.bin",
                "--host", "0.0.0.0", "--port", "8000"]
    if "/app" not in sys.path:
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
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
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
