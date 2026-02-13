"""
FlyWire Connectome Simulator — Web Server

Usage:
    pip install fastapi uvicorn[standard]
    python app_server.py                        # synthetic data
    python app_server.py --data flywire_v783.bin
"""

import argparse
import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sim_engine import SimEngine

# Global engines
torch_engine = None  # Lazy initialization for PyTorch engine
torch_setup_config = {}

parser = argparse.ArgumentParser(description="FlyWire Simulator Web Server")
parser.add_argument("--data", help="Binary connectome file")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

print("\nInitializing simulation engine...")
engine = SimEngine(data_file=args.data)
print(f"Engine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses\n")

app = FastAPI(title="FlyWire Simulator")

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

sim_running = False
batch_size = 200
clients: list[WebSocket] = []


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/positions")
async def get_positions():
    """Return neuron 3D positions + class info for the brain visualizer."""
    pos_b64 = engine.get_positions_b64()
    classes = engine.get_neuron_classes()
    return JSONResponse({
        "n_neurons": engine.n_neurons,
        "positions_b64": pos_b64,
        "classes": classes,
    })


@app.get("/io-analysis")
async def io_analysis_page():
    """Serve the I/O analysis page."""
    return FileResponse(os.path.join(static_dir, "io_analysis.html"))


@app.post("/api/io-analysis/setup")
async def setup_io_analysis(request: Request):
    """Setup custom input/output analysis.
    
    Expected request body:
    {
        "input_neurons": [{"id": 720575940629327659, "rate": 100}, ...],
        "output_neurons": [720575940629327659, 720575940604737708, ...]
    }
    
    Note: 'id' can be FlyWire root ID (large number) or neuron index (0-139254).
          'rate' is firing rate in Hz (spikes per second).
          Engine dt = 0.1 ms, so 100 Hz → 0.01 probability per timestep.
    """
    try:
        body = await request.json()
        input_neurons = body.get("input_neurons", [])
        output_neurons = body.get("output_neurons", [])
        
        # Set up rate-based stimulus for input neurons
        if input_neurons:
            engine.inject_stimulus_by_rate(input_neurons)
        else:
            engine.clear_stimulus()
        
        # Set up tracking for output neurons
        if output_neurons:
            engine.set_tracked_neurons(output_neurons)
            engine.reset_tracked_spike_counts()
        else:
            engine.clear_tracked_neurons()
        
        return JSONResponse({
            "success": True,
            "input_count": len(input_neurons) if input_neurons else 0,
            "output_count": len(output_neurons) if output_neurons else 0,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.post("/api/io-analysis/run")
async def run_io_analysis(request: Request):
    """Run simulation and get output neuron statistics.
    
    Expected request body:
    {
        "steps": 1000,
        "batch_size": 50
    }
    """
    try:
        body = await request.json()
        steps = body.get("steps", 1000)
        batch_size = body.get("batch_size", 50)
        
        # Run simulation
        total_steps = 0
        for i in range(0, steps, batch_size):
            n = min(batch_size, steps - i)
            engine.step(n=n)
            total_steps += n
        
        # Get statistics for tracked neurons
        stats = engine.get_tracked_neuron_stats(total_steps)
        
        return JSONResponse({
            "success": True,
            "steps_run": total_steps,
            "stats": stats
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.get("/io-analysis-torch")
async def io_analysis_torch_page():
    """Serve the PyTorch-based I/O analysis page."""
    return FileResponse(os.path.join(static_dir, "io_analysis_torch.html"))


@app.post("/api/io-analysis-torch/setup")
async def setup_io_analysis_torch(request: Request):
    """Setup custom input/output analysis using PyTorch/nourse_model.
    
    Expected request body:
    {
        "input_neurons": [{"id": 720575940629327659, "rate": 100}, ...],
        "output_neurons": [720575940629327659, 720575940604737708, ...]
    }
    
    Note: Uses nourse_model TorchModel with alpha synapses and refractory period.
          More biologically accurate but slower than CUDA FastFly.
    """
    global torch_engine, torch_setup_config
    
    try:
        # Lazy initialization of torch engine
        if torch_engine is None:
            from torch_engine import TorchSimEngine
            torch_engine = TorchSimEngine(dt=0.1)
        
        body = await request.json()
        input_neurons = body.get("input_neurons", [])
        output_neurons = body.get("output_neurons", [])
        
        # Store config for reset capability
        torch_setup_config = {
            "input_neurons": input_neurons,
            "output_neurons": output_neurons,
        }
        
        # Reset engine state
        torch_engine.reset()
        
        # Set up rate-based stimulus for input neurons
        if input_neurons:
            torch_engine.inject_stimulus_by_rate(input_neurons)
        
        # Set up tracking for output neurons
        if output_neurons:
            torch_engine.set_tracked_neurons(output_neurons)
        
        return JSONResponse({
            "success": True,
            "input_count": len(input_neurons) if input_neurons else 0,
            "output_count": len(output_neurons) if output_neurons else 0,
            "device": torch_engine.device,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.post("/api/io-analysis-torch/run")
async def run_io_analysis_torch(request: Request):
    """Run PyTorch simulation and get output neuron statistics.
    
    Expected request body:
    {
        "steps": 1000,
        "batch_size": 50
    }
    """
    global torch_engine
    print('Running PyTorch Simulation...') 
    try:
        if torch_engine is None:
            raise ValueError("Engine not initialized. Run setup first.")
        
        body = await request.json()
        steps = body.get("steps", 1000)
        batch_size = body.get("batch_size", 50)
        
        # Run simulation in batches
        total_steps = 0
        for i in range(0, steps, batch_size):
            n = min(batch_size, steps - i)
            torch_engine.step(n_steps=n)
            total_steps += n
        
        # Get statistics for tracked neurons
        stats = torch_engine.get_tracked_neuron_stats(total_steps)
        
        return JSONResponse({
            "success": True,
            "steps_run": total_steps,
            "stats": stats,
            "device": torch_engine.device,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


async def broadcast(msg: dict):
    data = json.dumps(msg)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


async def sim_loop():
    global sim_running
    while sim_running:
        try:
            metrics = await asyncio.get_event_loop().run_in_executor(
                None, engine.step, batch_size
            )
            metrics["type"] = "metrics"
            # Cap active_indices to limit WebSocket payload size
            ai = metrics.get("active_indices", [])
            if len(ai) > 5000:
                metrics["active_indices"] = ai[:5000]
            await broadcast(metrics)
            await asyncio.sleep(0)
        except Exception as e:
            print(f"sim_loop error: {e}", flush=True)
            sim_running = False
            await broadcast({"type": "state", "running": False})
            break


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global sim_running, batch_size

    await ws.accept()
    clients.append(ws)

    await ws.send_text(json.dumps({
        "type": "init",
        "n_neurons": engine.n_neurons,
        "n_synapses": engine.n_synapses,
        "stimuli": engine.get_predefined_stimuli(),
        "group_labels": engine.group_labels,
        "body_info": engine.get_body_info(),
    }))
    await ws.send_text(json.dumps({
        "type": "state",
        "running": sim_running,
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "start":
                if not sim_running:
                    sim_running = True
                    await broadcast({"type": "state", "running": True})
                    asyncio.create_task(sim_loop())

            elif cmd == "pause":
                sim_running = False
                await broadcast({"type": "state", "running": False})

            elif cmd == "step":
                sim_running = False
                metrics = await asyncio.get_event_loop().run_in_executor(
                    None, engine.step, batch_size
                )
                metrics["type"] = "metrics"
                await broadcast(metrics)

            elif cmd == "stimulus":
                indices = msg.get("indices", [])
                amplitude = float(msg.get("amplitude", 0.5))
                engine.inject_stimulus(indices, amplitude)

            elif cmd == "stimulus_preset":
                name = msg.get("name", "")
                amplitude = float(msg.get("amplitude", 0.5))
                engine.apply_predefined_stimulus(name, amplitude)

            elif cmd == "clear_stimulus":
                engine.clear_stimulus()

            elif cmd == "set_param":
                key = msg.get("key")
                value = msg.get("value")
                if key == "noise_amp":
                    engine.set_noise_amp(float(value))
                elif key == "batch_size":
                    batch_size = max(1, min(500, int(value)))
                elif key == "send_active_indices":
                    engine.send_active_indices = bool(value)
                elif key == "send_group_rates":
                    engine.send_group_rates = bool(value)
                elif key == "send_motor_rates":
                    engine.send_motor_rates = bool(value)

            elif cmd == "motor_detail":
                group_name = msg.get("group", "")
                detail = engine.get_motor_detail(group_name)
                if detail:
                    await ws.send_text(json.dumps({
                        "type": "motor_detail",
                        **detail,
                    }))

    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
