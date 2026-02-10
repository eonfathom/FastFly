"""
Download FlyWire neuron annotations and build a mapping from neuron index
(as used in flywire_v783.bin) to biological cell type, neuropil, nerve, etc.

Source: Schlegel et al. 2024, Nature - "Whole-brain annotation and
multi-connectome cell typing of Drosophila"

Output: neuron_annotations.npz
  - root_ids:     int64[N]   root_id for each neuron index
  - super_class:  U30[N]     e.g. 'sensory', 'motor', 'central', 'optic'
  - cell_class:   U60[N]     e.g. 'gustatory', 'olfactory', 'mechanosensory'
  - nerve:        U10[N]     e.g. 'AN' (antennal), 'CV' (cervical), 'MxLbN'
  - side:         U10[N]     'left' or 'right'
  - cell_type:    U80[N]     specific type e.g. 'ORN_DA1'
  - flow:         U15[N]     'afferent', 'efferent', 'intrinsic'
  - top_nt:       U15[N]     top neurotransmitter (e.g. 'acetylcholine')

Usage:
  pip install pandas requests
  python download_metadata.py
"""

import os
import sys
import numpy as np
import pandas as pd

ANNOTATION_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

COMPLETENESS_CSV = "Completeness_783.csv"
OUTPUT_FILE = "neuron_annotations.npz"
ANNOTATION_CACHE = "flywire_annotations_v783.tsv"


def main():
    print("=" * 64)
    print("  FlyWire Neuron Annotation Downloader")
    print("  Source: Schlegel et al. 2024 (Nature)")
    print("=" * 64)
    print()

    # Step 1: Load neuron index mapping from Completeness CSV
    if not os.path.exists(COMPLETENESS_CSV):
        print(f"ERROR: {COMPLETENESS_CSV} not found.")
        print("Run download_connectome.py first.")
        sys.exit(1)

    print("Step 1: Loading neuron index mapping...")
    df_neurons = pd.read_csv(COMPLETENESS_CSV, index_col=0)
    root_ids = df_neurons.index.values.astype(np.int64)
    n_neurons = len(root_ids)
    print(f"  {n_neurons} neurons in connectome")

    # Step 2: Download annotations
    print(f"\nStep 2: Downloading annotations...")
    if os.path.exists(ANNOTATION_CACHE):
        print(f"  Using cached: {ANNOTATION_CACHE}")
        df_ann = pd.read_csv(ANNOTATION_CACHE, sep='\t')
    else:
        print(f"  Downloading from GitHub...")
        try:
            df_ann = pd.read_csv(ANNOTATION_URL, sep='\t')
            df_ann.to_csv(ANNOTATION_CACHE, sep='\t', index=False)
            print(f"  Cached to {ANNOTATION_CACHE}")
        except Exception as e:
            print(f"  ERROR downloading: {e}")
            print(f"  URL: {ANNOTATION_URL}")
            sys.exit(1)

    print(f"  {len(df_ann)} annotated neurons")
    print(f"  Columns: {list(df_ann.columns)}")

    # Step 3: Match root_ids
    print(f"\nStep 3: Matching root_ids...")

    # Find the root_id column in annotations
    root_id_col = None
    for col in df_ann.columns:
        if 'root_id' in col.lower():
            root_id_col = col
            break
    if root_id_col is None:
        print(f"  ERROR: No root_id column found in annotations")
        print(f"  Columns: {list(df_ann.columns)}")
        sys.exit(1)

    print(f"  Root ID column: '{root_id_col}'")
    df_ann[root_id_col] = df_ann[root_id_col].astype(np.int64)

    # Build lookup from root_id -> annotation row
    ann_lookup = df_ann.set_index(root_id_col)

    # Columns we want to extract
    col_map = {
        'super_class': 'super_class',
        'cell_class': 'cell_class',
        'cell_sub_class': 'cell_sub_class',
        'cell_type': 'cell_type',
        'flow': 'flow',
        'nerve': 'nerve',
        'side': 'side',
        'top_nt': 'top_nt',
    }

    # Find matching columns (names might vary)
    actual_cols = {}
    for target, search in col_map.items():
        for col in ann_lookup.columns:
            if col.lower() == search.lower():
                actual_cols[target] = col
                break

    print(f"  Matched columns: {list(actual_cols.keys())}")

    # Build arrays
    arrays = {'root_ids': root_ids}
    for target, col in actual_cols.items():
        arr = []
        matched = 0
        for rid in root_ids:
            if rid in ann_lookup.index:
                val = ann_lookup.loc[rid, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.isna(val):
                    val = ''
                else:
                    val = str(val)
                arr.append(val)
                matched += 1
            else:
                arr.append('')
        arrays[target] = np.array(arr)
        print(f"  {target}: {matched}/{n_neurons} matched")

    # Step 3b: Extract 3D positions for brain visualization
    print(f"\nStep 3b: Extracting neuron 3D positions...")
    pos_x = np.zeros(n_neurons, dtype=np.float32)
    pos_y = np.zeros(n_neurons, dtype=np.float32)
    pos_z = np.zeros(n_neurons, dtype=np.float32)
    pos_matched = 0
    for i, rid in enumerate(root_ids):
        if rid in ann_lookup.index:
            row = ann_lookup.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            px = row.get('pos_x', np.nan)
            py = row.get('pos_y', np.nan)
            pz = row.get('pos_z', np.nan)
            if not (pd.isna(px) or pd.isna(py) or pd.isna(pz)):
                pos_x[i] = float(px)
                pos_y[i] = float(py)
                pos_z[i] = float(pz)
                pos_matched += 1
    print(f"  Positions matched: {pos_matched}/{n_neurons}")

    # Normalize to [-1, 1] centered at mean
    cx, cy, cz = pos_x.mean(), pos_y.mean(), pos_z.mean()
    pos_x -= cx; pos_y -= cy; pos_z -= cz
    scale = max(np.abs(pos_x).max(), np.abs(pos_y).max(), np.abs(pos_z).max())
    if scale > 0:
        pos_x /= scale; pos_y /= scale; pos_z /= scale
    print(f"  Normalized to [-1, 1], center=({cx:.0f}, {cy:.0f}, {cz:.0f}), scale={scale:.0f}")

    arrays['pos_x'] = pos_x
    arrays['pos_y'] = pos_y
    arrays['pos_z'] = pos_z

    # Step 4: Print summary statistics
    print(f"\nStep 4: Annotation summary...")

    if 'super_class' in arrays:
        sc = arrays['super_class']
        unique, counts = np.unique(sc[sc != ''], return_counts=True)
        print(f"\n  super_class distribution:")
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"    {u:<25} {c:>6}")

    if 'cell_class' in arrays:
        cc = arrays['cell_class']
        unique, counts = np.unique(cc[cc != ''], return_counts=True)
        print(f"\n  cell_class distribution (top 20):")
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:20]:
            print(f"    {u:<35} {c:>6}")

    if 'nerve' in arrays:
        nv = arrays['nerve']
        unique, counts = np.unique(nv[nv != ''], return_counts=True)
        print(f"\n  nerve distribution:")
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"    {u:<15} {c:>6}")

    # Step 5: Build stimulus groups
    print(f"\nStep 5: Building stimulus groups...")

    stimuli = {}
    sc = arrays.get('super_class', np.array([]))
    cc = arrays.get('cell_class', np.array([]))
    nv = arrays.get('nerve', np.array([]))
    sd = arrays.get('side', np.array([]))

    def find_neurons(mask):
        return np.where(mask)[0]

    # Gustatory neurons (proboscis/mouth) via labial/pharyngeal nerves
    gust_mask = (cc == 'gustatory')
    if gust_mask.any():
        # Left vs right
        gust_l = find_neurons(gust_mask & (sd == 'left'))
        gust_r = find_neurons(gust_mask & (sd == 'right'))
        if len(gust_l) > 0:
            stimuli['Sugar (left proboscis)'] = gust_l
            print(f"  Sugar (left proboscis):   {len(gust_l)} neurons")
        if len(gust_r) > 0:
            stimuli['Sugar (right proboscis)'] = gust_r
            print(f"  Sugar (right proboscis):  {len(gust_r)} neurons")
        stimuli['Sugar (proboscis)'] = find_neurons(gust_mask)
        print(f"  Sugar (proboscis):        {gust_mask.sum()} neurons")

    # Ascending neurons from legs (cervical connective)
    asc_mask = (sc == 'ascending')
    if asc_mask.any():
        asc_l = find_neurons(asc_mask & (sd == 'left'))
        asc_r = find_neurons(asc_mask & (sd == 'right'))
        if len(asc_l) > 0:
            stimuli['Touch (left leg/body)'] = asc_l
            print(f"  Touch (left leg/body):    {len(asc_l)} neurons")
        if len(asc_r) > 0:
            stimuli['Touch (right leg/body)'] = asc_r
            print(f"  Touch (right leg/body):   {len(asc_r)} neurons")

    # Mechanosensory via antennal nerve
    mech_mask = (cc == 'mechanosensory')
    if mech_mask.any():
        mech_l = find_neurons(mech_mask & (sd == 'left'))
        mech_r = find_neurons(mech_mask & (sd == 'right'))
        if len(mech_l) > 0:
            stimuli['Tickle (left antenna)'] = mech_l
            print(f"  Tickle (left antenna):    {len(mech_l)} neurons")
        if len(mech_r) > 0:
            stimuli['Tickle (right antenna)'] = mech_r
            print(f"  Tickle (right antenna):   {len(mech_r)} neurons")

    # Olfactory (smell)
    olf_mask = (cc == 'olfactory')
    if olf_mask.any():
        stimuli['Odor (both antennae)'] = find_neurons(olf_mask)
        print(f"  Odor (both antennae):     {olf_mask.sum()} neurons")
        olf_l = find_neurons(olf_mask & (sd == 'left'))
        olf_r = find_neurons(olf_mask & (sd == 'right'))
        if len(olf_l) > 0:
            stimuli['Odor (left antenna)'] = olf_l
            print(f"  Odor (left antenna):      {len(olf_l)} neurons")
        if len(olf_r) > 0:
            stimuli['Odor (right antenna)'] = olf_r
            print(f"  Odor (right antenna):     {len(olf_r)} neurons")

    # Thermosensory / hygrosensory
    thermo_mask = (cc == 'thermosensory') | (cc == 'thermo-hygrosensory')
    if thermo_mask.any():
        stimuli['Temperature change'] = find_neurons(thermo_mask)
        print(f"  Temperature change:       {thermo_mask.sum()} neurons")

    hygro_mask = (cc == 'hygrosensory')
    if hygro_mask.any():
        stimuli['Humidity change'] = find_neurons(hygro_mask)
        print(f"  Humidity change:          {hygro_mask.sum()} neurons")

    # Visual (optic lobe)
    vis_mask = (sc == 'visual_projection') | (sc == 'optic')
    if vis_mask.any():
        vis_l = find_neurons(vis_mask & (sd == 'left'))
        vis_r = find_neurons(vis_mask & (sd == 'right'))
        if len(vis_l) > 0:
            stimuli['Light (left eye)'] = vis_l
            print(f"  Light (left eye):         {len(vis_l)} neurons")
        if len(vis_r) > 0:
            stimuli['Light (right eye)'] = vis_r
            print(f"  Light (right eye):        {len(vis_r)} neurons")

    # Motor / descending (readout, not stimulus, but interesting)
    motor_mask = (sc == 'motor') | (sc == 'descending')
    if motor_mask.any():
        stimuli['Motor neurons (readout)'] = find_neurons(motor_mask)
        print(f"  Motor neurons (readout):  {motor_mask.sum()} neurons")

    # Step 6: Build neuropil-based groups for heatmap
    print(f"\nStep 6: Building neuropil groups for heatmap...")

    # Group neurons by super_class for meaningful heatmap rows
    heatmap_groups = {}
    for label in ['sensory', 'ascending', 'visual_projection', 'optic',
                   'central', 'descending', 'motor', 'endocrine',
                   'visual_centrifugal']:
        mask = (sc == label)
        if mask.any():
            heatmap_groups[label] = find_neurons(mask)
            print(f"  {label:<25} {mask.sum():>6} neurons")

    # Sub-divide 'central' which is huge — by side
    central_mask = (sc == 'central')
    if central_mask.any():
        central_l = find_neurons(central_mask & (sd == 'left'))
        central_r = find_neurons(central_mask & (sd == 'right'))
        if len(central_l) > 0:
            heatmap_groups['central_left'] = central_l
        if len(central_r) > 0:
            heatmap_groups['central_right'] = central_r
        # Remove the combined 'central' since we split it
        if 'central' in heatmap_groups:
            del heatmap_groups['central']

    # Step 6b: Build body-part motor/sensory groups for fly diagram
    print(f"\nStep 6b: Building body-part groups for fly diagram...")

    # --- Sensory input populations (body parts that can be stimulated) ---
    body_sensory = {}

    # Eyes (visual sensory neurons)
    vis_sensory = (cc == 'visual') | ((sc == 'optic') | (sc == 'visual_projection'))
    vis_l = find_neurons(vis_sensory & (sd == 'left'))
    vis_r = find_neurons(vis_sensory & (sd == 'right'))
    if len(vis_l): body_sensory['eye_left'] = vis_l
    if len(vis_r): body_sensory['eye_right'] = vis_r

    # Antennae - olfactory
    olf_l = find_neurons((cc == 'olfactory') & (sd == 'left'))
    olf_r = find_neurons((cc == 'olfactory') & (sd == 'right'))
    if len(olf_l): body_sensory['antenna_olf_left'] = olf_l
    if len(olf_r): body_sensory['antenna_olf_right'] = olf_r

    # Antennae - mechanosensory
    mech_l = find_neurons((cc == 'mechanosensory') & (sd == 'left'))
    mech_r = find_neurons((cc == 'mechanosensory') & (sd == 'right'))
    if len(mech_l): body_sensory['antenna_mech_left'] = mech_l
    if len(mech_r): body_sensory['antenna_mech_right'] = mech_r

    # Proboscis - gustatory
    gust_all = find_neurons(cc == 'gustatory')
    if len(gust_all): body_sensory['proboscis_gust'] = gust_all

    # Thermosensory / hygrosensory
    thermo_all = find_neurons((cc == 'thermosensory') | (cc == 'thermo-hygrosensory'))
    if len(thermo_all): body_sensory['thermo'] = thermo_all
    hygro_all = find_neurons(cc == 'hygrosensory')
    if len(hygro_all): body_sensory['hygro'] = hygro_all

    # Body/legs - ascending
    asc_l2 = find_neurons((sc == 'ascending') & (sd == 'left'))
    asc_r2 = find_neurons((sc == 'ascending') & (sd == 'right'))
    if len(asc_l2): body_sensory['body_left'] = asc_l2
    if len(asc_r2): body_sensory['body_right'] = asc_r2

    for k, v in body_sensory.items():
        print(f"  sensory {k:<25} {len(v):>5} neurons")

    # --- Motor output populations (body parts that respond) ---
    body_motor = {}

    motor_mask = (sc == 'motor')
    # Motor by nerve = body part
    motor_proboscis = find_neurons(motor_mask & (nv == 'MxLbN'))
    motor_pharynx = find_neurons(motor_mask & (nv == 'PhN'))
    motor_antenna = find_neurons(motor_mask & (nv == 'AN'))
    motor_neck = find_neurons(motor_mask & (nv == 'CV'))
    motor_eye = find_neurons(motor_mask & (nv == 'ON'))

    if len(motor_proboscis): body_motor['motor_proboscis'] = motor_proboscis
    if len(motor_pharynx):   body_motor['motor_pharynx'] = motor_pharynx
    if len(motor_antenna):   body_motor['motor_antenna'] = motor_antenna
    if len(motor_neck):      body_motor['motor_neck'] = motor_neck
    if len(motor_eye):       body_motor['motor_eye'] = motor_eye

    # Descending neurons by side (commands to legs/wings/body via VNC)
    desc_mask = (sc == 'descending')
    desc_l = find_neurons(desc_mask & (sd == 'left'))
    desc_r = find_neurons(desc_mask & (sd == 'right'))
    desc_c = find_neurons(desc_mask & (sd == 'center'))
    if len(desc_l): body_motor['descending_left'] = desc_l
    if len(desc_r): body_motor['descending_right'] = desc_r
    if len(desc_c): body_motor['descending_center'] = desc_c

    for k, v in body_motor.items():
        print(f"  motor   {k:<25} {len(v):>5} neurons")

    # Step 7: Save
    print(f"\nStep 7: Saving to {OUTPUT_FILE}...")

    save_dict = {}
    for key, arr in arrays.items():
        save_dict[key] = arr

    # Save stimuli as index arrays
    stim_names = []
    for name, indices in stimuli.items():
        safe_name = 'stim_' + name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        save_dict[safe_name] = indices.astype(np.int32)
        stim_names.append(name)
    save_dict['stim_names'] = np.array(stim_names)

    # Save heatmap groups
    group_names = []
    for name, indices in heatmap_groups.items():
        save_dict['group_' + name] = indices.astype(np.int32)
        group_names.append(name)
    save_dict['group_names'] = np.array(group_names)

    # Save body sensory groups
    body_sensory_names = []
    for name, indices in body_sensory.items():
        save_dict['bsens_' + name] = indices.astype(np.int32)
        body_sensory_names.append(name)
    save_dict['body_sensory_names'] = np.array(body_sensory_names)

    # Save body motor groups
    body_motor_names = []
    for name, indices in body_motor.items():
        save_dict['bmotor_' + name] = indices.astype(np.int32)
        body_motor_names.append(name)
    save_dict['body_motor_names'] = np.array(body_motor_names)

    np.savez_compressed(OUTPUT_FILE, **save_dict)
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"  Saved: {file_size / 1e6:.1f} MB")
    print(f"  Stimuli: {len(stim_names)}")
    print(f"  Heatmap groups: {len(group_names)}")
    print(f"  Body sensory: {len(body_sensory_names)}")
    print(f"  Body motor: {len(body_motor_names)}")

    print(f"\n{'='*64}")
    print(f"  SUCCESS: {OUTPUT_FILE}")
    print(f"  Use with: python app_server.py --data flywire_v783.bin")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
