"""
Download the real FlyWire connectome and convert to binary format
for the CUDA simulator.

Source: Shiu et al. 2024, Nature - "A Drosophila computational brain
model reveals sensorimotor processing"
GitHub: https://github.com/philshiu/Drosophila_brain_model

Data: Materialization v783 (final proofread version)
  - Connectivity_783.parquet: edge list with integer indices + signed weights
  - Completeness_783.csv: neuron metadata (defines index mapping)

Output: flywire_v783.bin (binary file for CUDA simulator)
  Header:  magic(4) + version(4) + num_neurons(4) + num_synapses(4)
  Data:    offsets[N+1] (uint32) + targets[S] (uint32) + weights[S] (float32)

Requirements: pip install pandas pyarrow numpy requests
"""

import os
import sys
import struct
import time
import numpy as np

def download_file(url, dest, desc=""):
    """Download a file with progress bar."""
    import urllib.request
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {desc or dest}...")

    def progress(count, block_size, total_size):
        mb = count * block_size / 1e6
        total_mb = total_size / 1e6 if total_size > 0 else 0
        if total_mb > 0:
            pct = min(100, mb / total_mb * 100)
            print(f"\r    {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\r    {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main():
    print("=" * 64)
    print("  FlyWire Connectome Downloader")
    print("  Source: Shiu et al. 2024 (Nature)")
    print("=" * 64)
    print()

    # --- Step 1: Download from GitHub ---
    print("Step 1: Downloading connectome data from GitHub...")

    # These are the raw GitHub URLs for the data files
    # The parquet file is stored with Git LFS, so we need the LFS URL
    GITHUB_BASE = "https://github.com/philshiu/Drosophila_brain_model"
    RAW_BASE = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"

    # For Git LFS files, we need to clone or use the media URL
    # Let's try direct raw first, fall back to clone
    parquet_file = "Connectivity_783.parquet"
    csv_file = "Completeness_783.csv"

    # Try downloading directly
    parquet_url = f"{RAW_BASE}/{parquet_file}"
    csv_url = f"{RAW_BASE}/{csv_file}"

    try:
        download_file(csv_url, csv_file, "neuron metadata (Completeness_783.csv)")
        download_file(parquet_url, parquet_file, "connectivity (Connectivity_783.parquet)")
    except Exception as e:
        print(f"\n  Direct download failed: {e}")
        print(f"\n  The parquet file may use Git LFS. Trying git clone...")
        print(f"  Running: git clone --depth 1 {GITHUB_BASE}")
        os.system(f'git clone --depth 1 {GITHUB_BASE} Drosophila_brain_model_repo')
        if os.path.exists(f"Drosophila_brain_model_repo/{parquet_file}"):
            import shutil
            shutil.copy(f"Drosophila_brain_model_repo/{parquet_file}", parquet_file)
            shutil.copy(f"Drosophila_brain_model_repo/{csv_file}", csv_file)
        else:
            print("\n  ERROR: Could not download the data files.")
            print(f"  Please manually clone: git clone {GITHUB_BASE}")
            print(f"  Then copy {parquet_file} and {csv_file} to this directory.")
            sys.exit(1)

    # --- Step 2: Load and process ---
    print("\nStep 2: Loading connectome data...")
    import pandas as pd

    df_neurons = pd.read_csv(csv_file, index_col=0)
    num_neurons = len(df_neurons)
    print(f"  Neurons: {num_neurons}")

    df_conn = pd.read_parquet(parquet_file)
    num_edges = len(df_conn)
    print(f"  Edges (synaptic connections): {num_edges}")
    print(f"  Columns: {list(df_conn.columns)}")

    # Extract arrays
    pre  = df_conn["Presynaptic_Index"].values.astype(np.int32)
    post = df_conn["Postsynaptic_Index"].values.astype(np.int32)

    # Use "Excitatory x Connectivity" — this is the signed weight from Shiu et al.
    # Positive = excitatory, negative = inhibitory, magnitude = synapse count
    # Do NOT use plain "Connectivity" (always positive, loses inhibitory sign)
    weight_col = "Excitatory x Connectivity"
    if weight_col not in df_conn.columns:
        print(f"  WARNING: '{weight_col}' not found.")
        print(f"  Available columns: {list(df_conn.columns)}")
        # Fall back to any column with both words
        for col in df_conn.columns:
            if "excitatory" in col.lower() and "connectivity" in col.lower():
                weight_col = col
                break
        else:
            print("  ERROR: Cannot find signed weight column.")
            sys.exit(1)

    print(f"  Weight column: '{weight_col}'")
    weights = df_conn[weight_col].values.astype(np.float32)

    # Validate
    assert pre.min() >= 0, f"Negative pre index: {pre.min()}"
    assert post.min() >= 0, f"Negative post index: {post.min()}"
    assert pre.max() < num_neurons, f"Pre index {pre.max()} >= {num_neurons}"
    assert post.max() < num_neurons, f"Post index {post.max()} >= {num_neurons}"

    print(f"\n  Pre-synaptic index range:  [{pre.min()}, {pre.max()}]")
    print(f"  Post-synaptic index range: [{post.min()}, {post.max()}]")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Excitatory: {(weights > 0).sum()} ({100*(weights>0).mean():.1f}%)")
    print(f"  Inhibitory: {(weights < 0).sum()} ({100*(weights<0).mean():.1f}%)")

    # --- Step 3: Build CSR format ---
    print("\nStep 3: Building CSR (Compressed Sparse Row) format...")
    t0 = time.time()

    # Sort by pre-synaptic neuron (row), then by post-synaptic (for cache locality)
    sort_idx = np.lexsort((post, pre))
    pre_sorted  = pre[sort_idx]
    post_sorted = post[sort_idx]
    wt_sorted   = weights[sort_idx]

    # Build offsets array
    offsets = np.zeros(num_neurons + 1, dtype=np.uint32)
    for i in range(num_edges):
        offsets[pre_sorted[i] + 1] += 1
    np.cumsum(offsets, out=offsets)

    assert offsets[-1] == num_edges, f"Offset mismatch: {offsets[-1]} vs {num_edges}"

    targets = post_sorted.astype(np.uint32)

    t1 = time.time()
    print(f"  CSR build time: {t1-t0:.1f}s")

    # Degree statistics
    degrees = np.diff(offsets)
    print(f"  Out-degree: min={degrees.min()}, max={degrees.max()}, "
          f"mean={degrees.mean():.1f}, median={np.median(degrees):.0f}")
    print(f"  Neurons with 0 out-degree: {(degrees == 0).sum()}")

    # --- Step 4: Normalize weights for LIF model ---
    print("\nStep 4: Normalizing weights for LIF simulation...")

    # Shiu et al. use: syn.w = weight * 0.275 * mV
    # For our LIF with threshold=1.0, we want weights such that
    # ~20-100 simultaneous excitatory inputs push a neuron to threshold.
    # With mean excitatory syn_count ~5-10, and 0.275mV per count,
    # that gives ~1.4-2.8 mV per connection.
    # Our threshold is 1.0 (unitless), so we scale:
    #   w_sim = raw_weight * scale_factor
    # where scale_factor makes the dynamics reasonable.

    # The raw weights are signed syn_counts (e.g., +7 means 7 excitatory synapses).
    # We want each synapse to contribute ~0.01-0.05 to the voltage.
    abs_mean = np.abs(wt_sorted).mean()
    if abs_mean > 0:
        scale = 0.03 / abs_mean  # target ~0.03 per average connection
    else:
        scale = 0.01
    wt_normalized = (wt_sorted * scale).astype(np.float32)

    print(f"  Raw weight |mean|: {abs_mean:.4f}")
    print(f"  Scale factor: {scale:.6f}")
    print(f"  Normalized weight range: [{wt_normalized.min():.6f}, {wt_normalized.max():.6f}]")
    print(f"  Normalized |mean|: {np.abs(wt_normalized).mean():.6f}")

    # --- Step 5: Write binary file ---
    output_file = "flywire_v783.bin"
    print(f"\nStep 5: Writing binary file '{output_file}'...")

    MAGIC = 0x464C5957  # "FLYW"
    VERSION = 1

    with open(output_file, "wb") as f:
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_neurons))
        f.write(struct.pack("<I", num_edges))

        # CSR offsets
        f.write(offsets.tobytes())

        # Targets
        f.write(targets.tobytes())

        # Weights (float32, will be converted to FP16 on GPU)
        f.write(wt_normalized.tobytes())

    file_size = os.path.getsize(output_file)
    print(f"  File size: {file_size / 1e6:.1f} MB")
    print(f"    Header:   16 bytes")
    print(f"    Offsets:  {(num_neurons+1)*4/1e6:.1f} MB")
    print(f"    Targets:  {num_edges*4/1e6:.1f} MB")
    print(f"    Weights:  {num_edges*4/1e6:.1f} MB")

    # --- Summary ---
    print(f"\n{'='*64}")
    print(f"  SUCCESS: {output_file}")
    print(f"  {num_neurons} neurons, {num_edges} synapses")
    print(f"  Ready for: flywire_sim.exe --data {output_file}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
