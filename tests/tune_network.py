#!/usr/bin/env python
"""Headless network tuning diagnostic.

Usage:
    python tests/tune_network.py --quick          # test current defaults
    python tests/tune_network.py --sweep          # grid search noise_amp x ei_ratio
    python tests/tune_network.py --quick --noise-amp 0.4 --ei-ratio 2.0
"""

import argparse
import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def make_engine(noise_amp=0.35, ei_ratio=2.0, weight_scale=1.0, seed=42):
    """Create a SimEngine with the given parameters."""
    from sim_engine import SimEngine
    engine = SimEngine(seed=seed, ei_ratio=ei_ratio, weight_scale=weight_scale,
                       noise_amp=noise_amp)
    return engine


def measure_baseline(engine, n_batches=20, batch_size=50, warmup_batches=5):
    """Run N batches with no stimulus, return firing rate + motor/command rates.

    Args:
        engine: SimEngine instance
        n_batches: Number of measurement batches (after warmup)
        batch_size: Steps per batch
        warmup_batches: Batches to discard before measuring

    Returns:
        dict with firing_rate, motor_rates, command_rates
    """
    engine.clear_stimulus()

    # Warmup — let transient die off
    for _ in range(warmup_batches):
        engine.step(n=batch_size)

    # Measure
    total_rate = 0.0
    motor_accum = {}
    command_accum = {}
    for _ in range(n_batches):
        m = engine.step(n=batch_size)
        total_rate += m["firing_rate"]
        for k, v in m.get("motor_rates", {}).items():
            motor_accum[k] = motor_accum.get(k, 0.0) + v
        for k, v in m.get("command_rates", {}).items():
            command_accum[k] = command_accum.get(k, 0.0) + v

    avg_rate = total_rate / n_batches
    avg_motor = {k: v / n_batches for k, v in motor_accum.items()}
    avg_command = {k: v / n_batches for k, v in command_accum.items()}

    return {
        "firing_rate": avg_rate,
        "motor_rates": avg_motor,
        "command_rates": avg_command,
    }


def measure_stimulus(engine, stimulus_fn, n_batches=20, batch_size=50,
                     warmup_batches=3):
    """Apply stimulus and measure response.

    Args:
        engine: SimEngine instance
        stimulus_fn: Callable(engine) that injects stimulus
        n_batches: Measurement batches
        batch_size: Steps per batch
        warmup_batches: Batches to let stimulus propagate

    Returns:
        dict with firing_rate, motor_rates, command_rates
    """
    engine.clear_stimulus()
    stimulus_fn(engine)

    # Warmup with stimulus active
    for _ in range(warmup_batches):
        # Re-apply ephemeral stimulus each batch
        stimulus_fn(engine)
        engine.step(n=batch_size)

    # Measure
    total_rate = 0.0
    motor_accum = {}
    command_accum = {}
    for _ in range(n_batches):
        stimulus_fn(engine)
        m = engine.step(n=batch_size)
        total_rate += m["firing_rate"]
        for k, v in m.get("motor_rates", {}).items():
            motor_accum[k] = motor_accum.get(k, 0.0) + v
        for k, v in m.get("command_rates", {}).items():
            command_accum[k] = command_accum.get(k, 0.0) + v

    avg_rate = total_rate / n_batches
    avg_motor = {k: v / n_batches for k, v in motor_accum.items()}
    avg_command = {k: v / n_batches for k, v in command_accum.items()}

    engine.clear_stimulus()

    return {
        "firing_rate": avg_rate,
        "motor_rates": avg_motor,
        "command_rates": avg_command,
    }


# --- Stimulus functions ---

def stim_left_odor(engine):
    """Strong attractive odor on left antenna."""
    engine.apply_environmental_scent(left_conc=0.8, right_conc=0.0)


def stim_bilateral_odor(engine):
    """Bilateral attractive odor."""
    engine.apply_environmental_scent(left_conc=0.6, right_conc=0.6)


def stim_current_injection(engine):
    """Direct current injection to first 1000 neurons."""
    indices = np.arange(0, min(1000, engine.n_neurons), dtype=np.int64)
    engine._apply_current_stimulus_additive(indices, 0.5)


def stim_gustatory(engine):
    """Sugar contact stimulus."""
    engine.apply_gustatory_stimulus(intensity=0.8)


STIMULI = {
    "left_odor": stim_left_odor,
    "bilateral_odor": stim_bilateral_odor,
    "current_1k": stim_current_injection,
    "gustatory": stim_gustatory,
}


def run_diagnostic(engine, n_batches=15, batch_size=50):
    """Run baseline + each stimulus type, print comparison table."""
    print("\n" + "=" * 70)
    print("  NETWORK DIAGNOSTIC")
    print("=" * 70)
    print(f"  Neurons: {engine.n_neurons:,}  Synapses: {engine.n_synapses:,}")
    print(f"  noise_amp: {float(engine.noise_amp):.3f}  "
          f"tau_decay: {float(engine.tau_decay):.2f}  "
          f"v_threshold: {float(engine.v_threshold):.2f}")
    print()

    # Baseline
    print("Measuring baseline (no stimulus)...")
    t0 = time.perf_counter()
    baseline = measure_baseline(engine, n_batches=n_batches, batch_size=batch_size)
    t1 = time.perf_counter()
    print(f"  Baseline firing rate: {baseline['firing_rate']*100:.3f}%  ({t1-t0:.1f}s)")

    # Stimuli
    results = {"baseline": baseline}
    for name, fn in STIMULI.items():
        print(f"Measuring stimulus: {name}...")
        t0 = time.perf_counter()
        res = measure_stimulus(engine, fn, n_batches=n_batches, batch_size=batch_size)
        t1 = time.perf_counter()
        delta = (res["firing_rate"] - baseline["firing_rate"])
        pct = (delta / baseline["firing_rate"] * 100) if baseline["firing_rate"] > 0 else float('inf')
        print(f"  {name}: {res['firing_rate']*100:.3f}%  "
              f"delta={delta*100:+.3f}%  ({pct:+.1f}% change)  ({t1-t0:.1f}s)")
        results[name] = res

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Condition':<20} {'Rate%':>8} {'Delta%':>8} {'Change':>8}")
    print("-" * 70)
    bl_rate = baseline["firing_rate"]
    print(f"{'baseline':<20} {bl_rate*100:>8.3f} {'---':>8} {'---':>8}")
    for name in STIMULI:
        r = results[name]["firing_rate"]
        d = r - bl_rate
        pct = (d / bl_rate * 100) if bl_rate > 0 else float('inf')
        print(f"{name:<20} {r*100:>8.3f} {d*100:>+8.3f} {pct:>+7.1f}%")
    print("-" * 70)

    # Motor rates comparison
    if baseline.get("motor_rates"):
        print(f"\n{'Motor Group':<25} {'Baseline':>10} {'Left Odor':>10} {'Gustatory':>10}")
        print("-" * 60)
        for k in sorted(baseline["motor_rates"].keys()):
            bl = baseline["motor_rates"].get(k, 0)
            lo = results.get("left_odor", {}).get("motor_rates", {}).get(k, 0)
            gu = results.get("gustatory", {}).get("motor_rates", {}).get(k, 0)
            print(f"{k:<25} {bl*100:>10.4f} {lo*100:>10.4f} {gu*100:>10.4f}")

    # Command neuron rates
    if baseline.get("command_rates"):
        print(f"\n{'Command Neuron':<20} {'Baseline':>10} {'Left Odor':>10} {'Gustatory':>10}")
        print("-" * 55)
        for k in sorted(baseline["command_rates"].keys()):
            bl = baseline["command_rates"].get(k, 0)
            lo = results.get("left_odor", {}).get("command_rates", {}).get(k, 0)
            gu = results.get("gustatory", {}).get("command_rates", {}).get(k, 0)
            print(f"{k:<20} {bl:>10.6f} {lo:>10.6f} {gu:>10.6f}")

    # Verdict
    print()
    if 0.01 <= bl_rate <= 0.05:
        print("PASS: Baseline in target range (1-5%)")
    elif bl_rate < 0.01:
        print(f"FAIL: Baseline too low ({bl_rate*100:.3f}%) — increase noise_amp or decrease ei_ratio")
    else:
        print(f"FAIL: Baseline too high ({bl_rate*100:.3f}%) — decrease noise_amp or increase ei_ratio")

    any_delta = any(
        results[name]["firing_rate"] > bl_rate * 1.1
        for name in STIMULI
    )
    if any_delta:
        print("PASS: At least one stimulus shows >10% increase over baseline")
    else:
        print("WARN: No stimulus produced >10% increase over baseline")

    return results


def run_sweep(seed=42):
    """Grid search over noise_amp and ei_ratio."""
    noise_amps = [0.2, 0.3, 0.35, 0.4, 0.5]
    ei_ratios = [1.8, 2.0, 2.1, 2.2, 2.33]
    n_batches = 10
    batch_size = 50

    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP: noise_amp x ei_ratio")
    print("=" * 70)
    print(f"  noise_amps: {noise_amps}")
    print(f"  ei_ratios:  {ei_ratios}")
    print(f"  {n_batches} batches x {batch_size} steps per measurement")
    print()

    # Header
    header = f"{'noise':>7} {'ei_rat':>7}"
    for label in ["base%", "stim%", "delta%", "change"]:
        header += f" {label:>8}"
    header += "  verdict"
    print(header)
    print("-" * 80)

    best = None
    best_score = -1

    for na in noise_amps:
        for ei in ei_ratios:
            t0 = time.perf_counter()
            try:
                engine = make_engine(noise_amp=na, ei_ratio=ei, seed=seed)
                bl = measure_baseline(engine, n_batches=n_batches,
                                      batch_size=batch_size, warmup_batches=5)
                stim = measure_stimulus(engine, stim_current_injection,
                                        n_batches=n_batches, batch_size=batch_size,
                                        warmup_batches=3)
            except Exception as e:
                print(f"{na:>7.2f} {ei:>7.2f}  ERROR: {e}")
                continue

            bl_rate = bl["firing_rate"]
            st_rate = stim["firing_rate"]
            delta = st_rate - bl_rate
            pct = (delta / bl_rate * 100) if bl_rate > 0 else 0

            # Score: want baseline 1-5% and good stimulus response
            in_range = 0.01 <= bl_rate <= 0.05
            score = 0
            if in_range:
                score += 10
                # Bonus for being near 3% (center of target)
                score += max(0, 5 - abs(bl_rate - 0.03) * 200)
            # Bonus for stimulus delta
            score += min(pct / 10, 5)

            verdict = "OK" if in_range else ("LOW" if bl_rate < 0.01 else "HIGH")
            if in_range and pct > 10:
                verdict = "GOOD"
            if in_range and pct > 50:
                verdict = "GREAT"

            elapsed = time.perf_counter() - t0
            print(f"{na:>7.2f} {ei:>7.2f} {bl_rate*100:>8.3f} {st_rate*100:>8.3f} "
                  f"{delta*100:>+8.3f} {pct:>+7.1f}%  {verdict:<6} ({elapsed:.1f}s)")

            if score > best_score:
                best_score = score
                best = {"noise_amp": na, "ei_ratio": ei,
                        "baseline": bl_rate, "stimulus": st_rate,
                        "delta_pct": pct, "verdict": verdict}

    print("-" * 80)
    if best:
        print(f"\nBest: noise_amp={best['noise_amp']}, ei_ratio={best['ei_ratio']}")
        print(f"  Baseline: {best['baseline']*100:.3f}%")
        print(f"  Stimulus: {best['stimulus']*100:.3f}%  ({best['delta_pct']:+.1f}% change)")
        print(f"\nTo apply, update sim_engine.py SimEngine.__init__ defaults:")
        print(f"  noise_amp={best['noise_amp']}, ei_ratio={best['ei_ratio']}")


def main():
    parser = argparse.ArgumentParser(
        description="Headless network tuning diagnostic")
    parser.add_argument("--quick", action="store_true",
                        help="Test current/specified params")
    parser.add_argument("--sweep", action="store_true",
                        help="Grid search noise_amp x ei_ratio")
    parser.add_argument("--noise-amp", type=float, default=None,
                        help="Override noise amplitude")
    parser.add_argument("--ei-ratio", type=float, default=None,
                        help="Override E/I ratio")
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="Override weight scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=15,
                        help="Measurement batches per condition")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Steps per batch")
    args = parser.parse_args()

    if not args.quick and not args.sweep:
        parser.print_help()
        print("\nSpecify --quick or --sweep")
        sys.exit(1)

    if args.sweep:
        run_sweep(seed=args.seed)
        return

    # --quick mode
    kwargs = {"seed": args.seed}
    if args.noise_amp is not None:
        kwargs["noise_amp"] = args.noise_amp
    if args.ei_ratio is not None:
        kwargs["ei_ratio"] = args.ei_ratio
    kwargs["weight_scale"] = args.weight_scale

    engine = make_engine(**kwargs)
    run_diagnostic(engine, n_batches=args.batches, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
