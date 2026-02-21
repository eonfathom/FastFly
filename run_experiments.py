"""
Phil's Experiments — P9 Locomotion + JO Auditory Input
======================================================

Experiment 1: P9_L + P9_R at 100 Hz (locomotion command only)
Experiment 2: P9_L + P9_R at 100 Hz + JO_E + JO_F at 150 Hz (locomotion + auditory)

Output: Downstream command neuron firing rates (Hz) in a comparison table.

Neuron IDs are loaded from static/neurons.json to match the connectome binary.

Usage:
    python run_experiments.py                          # default 200k steps
    python run_experiments.py --steps 50000            # quick test
    python run_experiments.py --data flywire_v783.bin  # explicit data file
"""

import argparse
import json
import os
import time


def load_neurons_json():
    """Load neuron definitions from static/neurons.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "static", "neurons.json")
    with open(path) as f:
        return json.load(f)


def get_single_id(neurons_data, name):
    """Get a single neuron root ID from neurons.json."""
    entry = neurons_data[name]
    rid = entry["id"]
    if isinstance(rid, list):
        raise ValueError(f"{name} has multiple IDs, expected single neuron")
    return int(rid)


def get_group_ids(neurons_data, name):
    """Get a list of neuron root IDs from a sensory group in neurons.json."""
    entry = neurons_data[name]
    rid_list = entry["id"]
    if not isinstance(rid_list, list):
        rid_list = [rid_list]
    ids = []
    for rid in rid_list:
        rid_str = str(rid).strip()
        if "," in rid_str:
            for sub in rid_str.split(","):
                ids.append(int(sub.strip()))
        else:
            ids.append(int(rid_str))
    return ids


def run_experiment(name, input_neurons, output_neurons, steps, data_file):
    """Run a single experiment and return output neuron firing rates in Hz."""
    from sim_engine import SimEngine

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    print("  Loading connectome...")
    t0 = time.perf_counter()
    engine = SimEngine(data_file=data_file)
    print(f"  Engine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses "
          f"({time.perf_counter()-t0:.1f}s)")

    # Set up input stimulus
    engine.inject_stimulus_by_rate(input_neurons)
    rates_set = sorted(set(nr["rate"] for nr in input_neurons))
    print(f"  Input neurons: {len(input_neurons)} at rates {rates_set} Hz")

    # Set up output tracking
    output_ids = list(output_neurons.values())
    engine.set_tracked_neurons(output_ids)
    engine.reset_tracked_spike_counts()

    # Run simulation
    batch = 500
    print(f"  Running {steps} steps ({steps * 0.1 / 1000:.1f}s bio time)...")
    t0 = time.perf_counter()
    for i in range(0, steps, batch):
        n = min(batch, steps - i)
        engine.step(n=n)
        if (i // batch) % 20 == 0:
            pct = 100 * (i + n) / steps
            elapsed = time.perf_counter() - t0
            print(f"    {pct:5.1f}%  ({elapsed:.1f}s)", end="\r")
    elapsed = time.perf_counter() - t0
    print(f"  Done: {steps} steps in {elapsed:.1f}s "
          f"({steps/elapsed:.0f} steps/s)")

    # Get results
    stats = engine.get_tracked_neuron_stats(steps)
    spike_counts = stats["spike_counts"]
    firing_rates_per_step = stats["firing_rates"]

    # Convert firing_rate_per_step to Hz: Hz = rate_per_step * (1000 / dt_ms)
    dt_ms = 0.1
    hz_scale = 1000.0 / dt_ms  # 10000

    results = {}
    for i, (label, root_id) in enumerate(output_neurons.items()):
        hz = firing_rates_per_step[i] * hz_scale
        results[label] = {
            "root_id": root_id,
            "spikes": spike_counts[i],
            "hz": hz,
        }

    return results


def print_comparison(results1, results2, name1, name2, output_neurons):
    """Print a comparison table of two experiment results."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")
    hdr = f"  {'Neuron':<12} {'FlyWire Root ID':>20}  {'Spk1':>7} {'Hz1':>8}  {'Spk2':>7} {'Hz2':>8}  {'Delta':>8}"
    print(hdr)
    print(f"  {'-'*len(hdr.strip())}")

    for label in output_neurons:
        r1 = results1[label]
        r2 = results2[label]
        delta = r2["hz"] - r1["hz"]
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<12} {r1['root_id']:>20}  "
              f"{r1['spikes']:>7} {r1['hz']:>8.2f}  "
              f"{r2['spikes']:>7} {r2['hz']:>8.2f}  "
              f"{sign}{delta:>7.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Phil's P9 + JO Experiments")
    parser.add_argument("--data", default="flywire_v783.bin",
                        help="Binary connectome file (default: flywire_v783.bin)")
    parser.add_argument("--steps", type=int, default=200000,
                        help="Simulation steps per experiment (default: 200000 = 20s bio)")
    args = parser.parse_args()

    print(f"Data file: {args.data}")
    print(f"Steps per experiment: {args.steps} ({args.steps * 0.1 / 1000:.1f}s bio time)")

    # Load neuron definitions from neurons.json
    neurons_data = load_neurons_json()

    # ── Input neurons ──────────────────────────────────────────────────
    p9_l_id = get_single_id(neurons_data, "P9_L")
    p9_r_id = get_single_id(neurons_data, "P9_R")
    jo_e_ids = get_group_ids(neurons_data, "JO_E")
    jo_f_ids = get_group_ids(neurons_data, "JO_F")

    print(f"\nInput neurons resolved from neurons.json:")
    print(f"  P9_L:  {p9_l_id}")
    print(f"  P9_R:  {p9_r_id}")
    print(f"  JO_E:  {len(jo_e_ids)} neurons")
    print(f"  JO_F:  {len(jo_f_ids)} neurons")

    # ── Output neurons (command neurons from neurons.json) ─────────────
    output_neurons = {
        "P9_L":    get_single_id(neurons_data, "P9_L"),
        "P9_R":    get_single_id(neurons_data, "P9_R"),
        "DNa02_L": get_single_id(neurons_data, "DNa02_L"),
        "DNa02_R": get_single_id(neurons_data, "DNa02_R"),
        "oDN1_L":  get_single_id(neurons_data, "oDN1_L"),
        "oDN1_R":  get_single_id(neurons_data, "oDN1_R"),
        "aDN1_L":  get_single_id(neurons_data, "aDN1_L"),
        "aDN1_R":  get_single_id(neurons_data, "aDN1_R"),
    }

    print(f"\nOutput neurons (8 command neurons):")
    for label, rid in output_neurons.items():
        print(f"  {label:<12} {rid}")

    # ── Experiment 1: P9 only ──────────────────────────────────────────
    input1 = [
        {"id": p9_l_id, "rate": 100},
        {"id": p9_r_id, "rate": 100},
    ]

    results1 = run_experiment(
        name="Exp 1: P9_L + P9_R @ 100 Hz",
        input_neurons=input1,
        output_neurons=output_neurons,
        steps=args.steps,
        data_file=args.data,
    )

    # ── Experiment 2: P9 + JO_E + JO_F ────────────────────────────────
    input2 = (
        [{"id": p9_l_id, "rate": 100}, {"id": p9_r_id, "rate": 100}] +
        [{"id": jid, "rate": 150} for jid in jo_e_ids] +
        [{"id": jid, "rate": 150} for jid in jo_f_ids]
    )

    results2 = run_experiment(
        name="Exp 2: P9 @ 100 Hz + JO_E/JO_F @ 150 Hz",
        input_neurons=input2,
        output_neurons=output_neurons,
        steps=args.steps,
        data_file=args.data,
    )

    # ── Comparison ─────────────────────────────────────────────────────
    print_comparison(
        results1, results2,
        "Exp 1 (P9 only)",
        "Exp 2 (P9 + JO_E/F)",
        output_neurons,
    )


if __name__ == "__main__":
    main()
