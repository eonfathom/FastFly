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

# ---------------------------------------------------------------------------
# 1. Container image — built once, cached by Modal
#    Split into layers so each caches independently.
# ---------------------------------------------------------------------------

# Layer 1: Base OS + CUDA toolkit (needed for CuPy kernel compilation)
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04",
    add_python="3.11",
)

# Layer 2: Core Python packages (cached after first build)
packages_image = cuda_image.pip_install(
    "cupy-cuda12x>=13.6.0",
    "fastapi>=0.129.0",
    "numpy>=1.24.0,<2.0.0",
    "uvicorn[standard]>=0.32.0",
    "pandas>=2.0.0",
    "pyarrow>=10.0.0",
    "requests",
)

# Layer 3: Clone repo + download connectome data
image = (
    packages_image
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/AqeelAqeel/FastFly.git /app",
        "cd /app && git checkout feat-glb-flybody-aqeel",
    )
    .run_commands(
        "cd /app && python download_connectome.py",
        "cd /app && python download_metadata.py",
    )
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
    import sys
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
    import subprocess
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
