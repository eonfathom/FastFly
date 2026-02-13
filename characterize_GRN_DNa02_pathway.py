# -*- coding: utf-8 -*-
"""
Characterize GRN to DNa02 Pathway Response

This script systematically tests the response of DNa02 neurons to different
combinations of left and right sugar GRN stimulation rates. It creates a
response matrix showing how DNa02 firing rates depend on GRN inputs.

The script runs independently of the physics simulation to purely characterize
the neural pathway properties.

Output: CSV file with columns [left_grn_rate, right_grn_rate, left_dna02_rate, right_dna02_rate]
        Can be interrupted and resumed - skips already-tested combinations
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add nourse_model to path
current_dir = os.path.dirname(__file__)
nourse_path = os.path.join(current_dir, 'nourse_model', 'src')
sys.path.insert(0, nourse_path)

import models as nourse_models
import utils as nourse_utils


# ===== CONFIGURATION =====
CONNECTOME_DT = 0.1  # ms - internal timestep
SIMULATION_DURATION = 500  # ms - how long to run for each test
NUM_STEPS = int(SIMULATION_DURATION / CONNECTOME_DT)

# Input stimulation mode
# 'single' - vary one input group only (INPUT_NEURONS_A)
# 'dual' - vary two independent input groups (INPUT_NEURONS_A and INPUT_NEURONS_B)
# 'triple' - vary three independent input groups (INPUT_NEURONS_A, B, and C)
INPUT_MODE = 'triple'  # Options: 'single', 'dual', or 'triple'

# Input stimulation rate ranges
# For INPUT_NEURONS_A
MIN_RATE_A = 0.0  # Hz
MAX_RATE_A = 200.0  # Hz
STEP_SIZE_A = 100.0  # Hz
RATES_A = np.arange(MIN_RATE_A, MAX_RATE_A + STEP_SIZE_A, STEP_SIZE_A)

# For INPUT_NEURONS_B
MIN_RATE_B = 0.0  # Hz
MAX_RATE_B = 96.0  # Hz
STEP_SIZE_B = 12.0  # Hz
RATES_B = np.arange(MIN_RATE_B, MAX_RATE_B + STEP_SIZE_B, STEP_SIZE_B)

# For INPUT_NEURONS_C
MIN_RATE_C = 0.0  # Hz
MAX_RATE_C = 96.0  # Hz
STEP_SIZE_C = 12.0  # Hz
RATES_C = np.arange(MIN_RATE_C, MAX_RATE_C + STEP_SIZE_C, STEP_SIZE_C)

# Output files
OUTPUT_CSV = 'triple_groom_forage_gpu.csv'
OUTPUT_PLOT = 'triple_groom_forage_gpu.png'

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Neuron IDs (from FlyWire dataset)
# Sugar GRNs - split by hemisphere
LEFT_SUGAR_GRNS = [
     720575940622486922,720575940611875570,720575940639259967,720575940616811265,720575940613601698,720575940607347634,720575940637568838,720575940617857694,720575940621754367,720575940622825736,720575940629176663,720575940625203504,720575940620926234,720575940638202345,720575940632889389,720575940609476562,720575940617000768,720575940632510479,720575940612579053,720575940630797113,720575940623172843,720575940630553415,720575940639332736,720575940633143833,720575940615261073,720575940616177458,720575940638183652,720575940610788069,720575940628853239,720575940639043280
]
RIGHT_SUGAR_GRNS = [
    720575940617398502, 720575940618601782, 720575940618706043, 720575940620296641, 720575940606205897, 720575940618706811, 720575940608305161, 720575940632627660, 720575940612208406, 720575940632634911, 720575940620296641, 720575940624373483, 720575940616167218, 720575940629388135, 720575940609314243, 720575940606801282, 720575940608305161, 720575940630882039, 720575940630552151, 720575940612612581, 720575940629510338, 720575940622136022, 720575940631147148, 720575940634023961, 720575940604590048, 720575940630968335, 720575940614302370, 720575940631656504, 720575940641339611, 720575940620589838
]

LEFT_ATTRACTIVE_ORNS = [ 720575940631174431, 720575940620028080, 720575940636052080, 720575940617681822, 720575940625757053, 720575940619093524, 720575940646142004,720575940616486795, 720575940631203896, 720575940638852736, 720575940640027637, 720575940611369938, 720575940639448528,720575940620319329,720575940646540910,720575940618512331, 720575940606108862, 720575940626571592, 720575940629232492, 720575940641372112, 720575940620572933, 720575940627737073, 720575940610999794, 720575940626530564, 720575940626879484, 720575940623482355, 720575940634530155, 720575940614257986, 720575940632315106, 720575940632356066, 720575940614833567, 720575940622833577, 720575940644076398, 720575940620822049
]

RIGHT_ATTRACTIVE_ORNS = [
    720575940638410611, 720575940645214243, 720575940646120324, 720575940616205405, 720575940613666666, 720575940645576756, 720575940624096708, 720575940645211939, 720575940607811420, 720575940613509010, 720575940614014431, 720575940622063871, 720575940628641871, 720575940633668031, 720575940610922202, 720575940619360532,720575940620555252, 720575940629263872, 720575940644028452, 720575940611237034, 720575940620030384, 720575940621326874, 720575940621821004, 720575940620831265, 720575940629677655, 720575940605834626, 720575940623801276, 720575940616194397, 720575940621808227, 720575940611218418, 720575940617009638, 720575940625383299, 720575940637846069
]

LEFT_AVOIDANT_ORNS = [720575940631174431, 720575940620028080,720575940636052080, 720575940617681822, 720575940625757053,720575940619093524,720575940646142004,720575940616486795,720575940631203896,720575940638852736,720575940640027637,720575940611369938,720575940639448528,720575940620319329,720575940646540910,720575940618512331,720575940606108862,720575940626571592,720575940629232492,720575940641372112,720575940620572933,720575940627737073,720575940610999794,720575940626530564,720575940626879484,720575940623482355, 720575940634530155, 720575940614257986, 720575940632315106, 720575940632356066, 720575940614833567, 720575940622833577,720575940644076398,720575940620822049];

RIGHT_AVOIDANT_ORNS = [720575940638410611, 720575940645214243, 720575940646120324, 720575940616205405, 720575940613666666, 720575940645576756,720575940624096708, 720575940645211939, 720575940607811420, 720575940613509010, 720575940614014431, 720575940622063871, 720575940628641871, 720575940633668031, 720575940610922202, 720575940619360532, 720575940620555252, 720575940629263872, 720575940644028452, 720575940611237034, 720575940620030384, 720575940621326874, 720575940621821004, 720575940620831265, 720575940629677655, 720575940605834626, 720575940623801276, 720575940616194397, 720575940621808227, 720575940611218418, 720575940617009638, 720575940625383299, 720575940637846069];

JO_C_NEURONS = [720575940622486922,720575940611875570,720575940639259967,720575940616811265,720575940613601698,720575940607347634,720575940637568838,720575940617857694,720575940621754367,720575940622825736,720575940629176663,720575940625203504,720575940620926234,720575940638202345,720575940632889389,720575940609476562,720575940617000768,720575940632510479,720575940612579053,720575940630797113,720575940623172843,720575940630553415,720575940639332736,720575940633143833,720575940615261073,720575940616177458,720575940638183652,720575940610788069,720575940628853239,720575940639043280]

JO_F_NEURONS = [720575940622357740,720575940624496280,720575940619800001,720575940617613392,720575940626200457,720575940631406667,720575940609547549, 720575940621593972,720575940630530263,720575940638142953,720575940605691424,720575940607452594,720575940615230175,720575940624123645, 720575940614529186,720575940608491019,720575940617212929,720575940629014119,720575940604901310,720575940615455230,720575940617145300,720575940625060280,720575940625966350,720575940630962280,720575940638312270,720575940643000480,720575940621051501,720575940622378281, 720575940623222271,720575940624595961,720575940632361111,720575940633957271,720575940636475511,720575940608399452,720575940611121362,720575940621563892,720575940622927242,720575940628941762,720575940630625132,720575940632683212,720575940633626082,720575940646704692, 720575940613331693,720575940626983293,720575940634612843,720575940634837343,720575940638144233,720575940611939314,720575940613036914,720575940614684834,720575940619697624,720575940623360394,720575940625003964,720575940625627024,720575940626606594,720575940627372184,720575940631970144,720575940610213785,720575940612375525,720575940615096495,720575940624761075,720575940628873295,720575940629032015,720575940630425285,720575940632170065,720575940638107445,720575940639090395,720575940611057506,720575940614326946,720575940623543236,720575940627099006,720575940629302016,720575940621005677,720575940624334827,720575940624903757,720575940625070067,720575940630472727,720575940604781158,720575940604810668,720575940617654818,720575940617967188,720575940619210968,720575940626149578,720575940626570648,720575940627327938,720575940633293538,720575940634683488,720575940637128538,720575940607544329,720575940612567449,720575940618558769,720575940620113799,720575940623979959,720575940634104159,720575940635159659,720575940636075119,720575940647034420,720575940619416881,720575940631108492,720575940618003284,720575940624846835,720575940628539335,720575940644611016,720575940621020257,720575940617241659,720575940622171169]

JO_E_NEURONS = [720575940605860710,720575940613153320,720575940618898230,720575940623412360,720575940631100690,720575940620293652,720575940630458732, 720575940621981464,720575940635028634,720575940645483555,720575940629706496,720575940619518667,720575940619557937,720575940632188497, 720575940639741347,720575940627542769,720575940628915279,720575940616793300,720575940617341270,720575940619235920,720575940619694040, 720575940621244510,720575940622163020,720575940628162300,720575940628908670,720575940630650130,720575940631367520,720575940631997580,720575940634419040,720575940638064100,720575940605929161,720575940613142191,720575940615276661,720575940616190651,720575940616479121,720575940616631001,720575940623334891,720575940626169481,720575940626606211,720575940631326751,720575940633810111,720575940634186091,720575940634605531,720575940634640091,720575940635290231,720575940609832772,720575940615926982,720575940619991232,720575940621590132,720575940622281352,720575940623928812,720575940626748232,720575940628210272,720575940605997233,720575940607651083,720575940609065173,720575940613803293,720575940615957563,720575940616642363,720575940618861873,720575940621639133,720575940634599643,720575940636332663, 720575940636768613,720575940637445183,720575940639041443,720575940639233333,720575940641656283,720575940618646744,720575940618896624, 720575940618927024,720575940623406984,720575940625293084,720575940625489764,720575940628306784,720575940631367264,720575940631368544,720575940631406944,720575940631957344,720575940634181984,720575940638332494,720575940604879025,720575940608658205,720575940609221845,720575940613428945,720575940614109615,720575940616162235,720575940618051485,720575940619930965,720575940620661675,720575940621172545,720575940622099405,720575940624115785,720575940626524805,720575940627925865,720575940629531095,720575940630094325,720575940632154285,720575940633555425,720575940634714075,720575940636290935,720575940638243645,720575940640651325,720575940640748195,720575940606912386,720575940608369026,720575940612136666,720575940618293406,720575940619235486,720575940620114196,720575940621136816,720575940622432726,720575940623001736,720575940624137656,720575940626270236,720575940628349446,720575940628355846,720575940631314316,720575940631367776,720575940631370336,720575940631412576,720575940636395866,720575940638571486,720575940651360246,720575940611197107,720575940617504027,720575940620055617,720575940621431247,720575940623300137,720575940628495447,720575940630964187,720575940631191367,720575940635672677,720575940636498907,720575940639298237,720575940640952027,720575940604976318,720575940610702968,720575940611029858,720575940617245908,720575940620791028,720575940622202858,720575940626543768,720575940628828688,720575940630174268,720575940635579128,720575940637022298,720575940611000499,720575940613309359,720575940613521389,720575940616345049,720575940616661109,720575940624157769,720575940626753809,720575940629114959,720575940630715899,720575940631044059,720575940631694069,720575940632475859,720575940635345519,720575940635959919,720575940636410869,720575940638431029,720575940638546229,720575940639699059,720575940612341490,720575940612841330,720575940616545110,720575940635136880,720575940639166080,720575940612279731,720575940619912021,720575940642309261,720575940614656162,720575940615090582,720575940621032513,720575940639648573,720575940612679594,720575940620537734,720575940622683174,720575940611986545,720575940623291945,720575940626881625,720575940615239746,720575940620538246,720575940608862677,720575940619218078,720575940620087888,720575940624664468,720575940629534458,720575940608986069,720575940612286899,720575940615301599,720575940633738209,720575940639161909]
########################################
# Configure input neurons based on mode
# For 'triple' mode: A=JO neurons, B=LEFT_GRNs, C=RIGHT_GRNs
INPUT_NEURONS_A = JO_E_NEURONS + JO_C_NEURONS + JO_F_NEURONS  # JO neurons (E+C+F)
INPUT_NEURONS_B = LEFT_SUGAR_GRNS  # Left GRNs
INPUT_NEURONS_C = RIGHT_SUGAR_GRNS  # Right GRNs
########################################

# Output Neurons - list-based structure for easy extension
OUTPUT_NEURONS = {
    'DNa02_L': 720575940629327659,
    'DNa02_R': 720575940604737708,
    'oDN1_L': 720575940626730883,
    'oDN1_R': 720575940620300308,
    'aDN1_L': 720575940624319124,
    'aDN1_R': 720575940616185531,
}

# P9 neurons (speed control) - constant stimulation at 100Hz
P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

print(f"Neural Pathway Characterization")
print(f"=" * 60)
print(f"Input mode: {INPUT_MODE}")
print(f"Device: {DEVICE}")
print(f"Connectome timestep: {CONNECTOME_DT} ms")
print(f"Simulation duration: {SIMULATION_DURATION} ms ({NUM_STEPS} steps)")
if INPUT_MODE == 'triple':
    print(f"Input A (JO) rate range: {MIN_RATE_A}-{MAX_RATE_A} Hz in {STEP_SIZE_A} Hz steps ({len(RATES_A)} values)")
    print(f"Input B (Left GRN) rate range: {MIN_RATE_B}-{MAX_RATE_B} Hz in {STEP_SIZE_B} Hz steps ({len(RATES_B)} values)")
    print(f"Input C (Right GRN) rate range: {MIN_RATE_C}-{MAX_RATE_C} Hz in {STEP_SIZE_C} Hz steps ({len(RATES_C)} values)")
    print(f"Total combinations to test: {len(RATES_A)} x {len(RATES_B)} x {len(RATES_C)} = {len(RATES_A) * len(RATES_B) * len(RATES_C)}")
elif INPUT_MODE == 'dual':
    print(f"Input A rate range: {MIN_RATE_A}-{MAX_RATE_A} Hz in {STEP_SIZE_A} Hz steps ({len(RATES_A)} values)")
    print(f"Input B rate range: {MIN_RATE_B}-{MAX_RATE_B} Hz in {STEP_SIZE_B} Hz steps ({len(RATES_B)} values)")
    print(f"Total combinations to test: {len(RATES_A)} x {len(RATES_B)} = {len(RATES_A) * len(RATES_B)}")
else:
    print(f"Input rate range: {MIN_RATE_A}-{MAX_RATE_A} Hz in {STEP_SIZE_A} Hz steps")
    print(f"Total rates to test: {len(RATES_A)}")
print(f"Output file: {OUTPUT_CSV}")
print(f"=" * 60)


def load_connectome():
    """Load the Nourse connectome model and neuron mappings."""
    print("\nLoading connectome model...")
    
    # Configure paths to data files
    data_dir = os.path.join(current_dir, 'nourse_model', 'data')
    path_comp = os.path.join(data_dir, 'Completeness_783.csv')
    path_conn = os.path.join(data_dir, 'Connectivity_783.parquet')
    path_wt = data_dir
    
    # Get neuron ID mappings
    flyid2i, i2flyid = nourse_utils.get_hash_tables(path_comp)
    
    # Load weights
    weights = nourse_utils.get_weights(path_conn, path_comp, path_wt)
    
    # Model parameters (from walkingFly_Nourse.py)
    model_params = {
        'tauSyn': 5.0,
        'tDelay': 1.8,
        'v0': -52.0,
        'vReset': -52.0,
        'vRest': -52.0,
        'vThreshold': -45.0,
        'tauMem': 20.0,
        'tRefrac': 2.2,
        'scalePoisson': 250,
        'wScale': 0.275,
    }
    
    # Create the model
    num_neurons = weights.shape[0]
    model = nourse_models.TorchModel(
        batch=1,
        size=num_neurons,
        dt=CONNECTOME_DT,
        params=model_params,
        weights=weights.to(device=DEVICE),
        device=DEVICE
    )
    
    print(f"Loaded connectome with {num_neurons} neurons")
    
    return model, flyid2i, i2flyid, num_neurons


def initialize_state(model, num_neurons):
    """Initialize connectome state variables."""
    conductance, delay_buffer, spikes, v, refrac = model.state_init()
    rates = torch.zeros((1, num_neurons), device=DEVICE)
    
    return conductance, delay_buffer, spikes, v, refrac, rates


def run_simulation(model, flyid2i, input_a_rate, input_b_rate, input_c_rate=0.0):
    """
    Run connectome simulation with specified input neuron stimulation rates.
    
    Args:
        model: The connectome model
        flyid2i: Neuron ID to index mapping
        input_a_rate: Stimulation rate for INPUT_NEURONS_A (Hz)
        input_b_rate: Stimulation rate for INPUT_NEURONS_B (Hz)
        input_c_rate: Stimulation rate for INPUT_NEURONS_C (Hz) - optional
    
    Returns dictionary of firing rates (Hz) for all output neurons.
    """
    num_neurons = len(flyid2i)
    
    # Initialize state
    conductance, delay_buffer, spikes, v, refrac, rates = initialize_state(model, num_neurons)
    
    # Set stimulation rates
    rates.zero_()
    
    # Stimulate P9 neurons at constant 100Hz (for consistency)
    p9_left_idx = flyid2i[P9_LEFT]
    p9_right_idx = flyid2i[P9_RIGHT]
    rates[0, p9_left_idx] = 100.0
    rates[0, p9_right_idx] = 100.0
    
    # Stimulate input neurons group A
    if input_a_rate > 0 and len(INPUT_NEURONS_A) > 0:
        for nid in INPUT_NEURONS_A:
            rates[0, flyid2i[nid]] = input_a_rate
    
    # Stimulate input neurons group B
    if input_b_rate > 0 and len(INPUT_NEURONS_B) > 0:
        for nid in INPUT_NEURONS_B:
            rates[0, flyid2i[nid]] = input_b_rate
    
    # Stimulate input neurons group C (only in triple mode)
    if input_c_rate > 0 and len(INPUT_NEURONS_C) > 0:
        for nid in INPUT_NEURONS_C:
            rates[0, flyid2i[nid]] = input_c_rate
    
    # Run simulation
    times_list = torch.tensor([], device=DEVICE)
    idx_list = torch.tensor([], device=DEVICE)
    
    with torch.no_grad():
        for t in range(NUM_STEPS):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )
            # Track spikes
            times_list, idx_list = nourse_utils.get_spike_times(
                spikes[0, :], t, CONNECTOME_DT, times_list, idx_list
            )
    
    # Count spikes for all output neurons
    output_spike_counts = {name: 0 for name in OUTPUT_NEURONS.keys()}
    
    if len(idx_list) > 0:
        idx_cpu = idx_list.cpu().numpy()
        for name, neuron_id in OUTPUT_NEURONS.items():
            neuron_idx = flyid2i[neuron_id]
            output_spike_counts[name] = np.sum(idx_cpu == neuron_idx)
    
    # Convert to firing rates (Hz)
    duration_s = SIMULATION_DURATION / 1000.0
    output_rates = {name: count / duration_s for name, count in output_spike_counts.items()}
    
    return output_rates


def load_existing_results():
    """Load existing results if the CSV file exists."""
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\nLoaded {len(df)} existing results from {OUTPUT_CSV}")
        return df
    else:
        print(f"\nNo existing results found. Starting fresh.")
        # Create columns dynamically based on OUTPUT_NEURONS and INPUT_MODE
        if INPUT_MODE == 'triple':
            columns = ['input_a_rate', 'input_b_rate', 'input_c_rate'] + list(OUTPUT_NEURONS.keys())
        else:
            columns = ['input_a_rate', 'input_b_rate'] + list(OUTPUT_NEURONS.keys())
        return pd.DataFrame(columns=columns)


def save_result(df, input_a_rate, input_b_rate, output_rates, input_c_rate=0.0):
    """Append a single result to the dataframe and save to CSV."""
    new_row = {
        'input_a_rate': input_a_rate,
        'input_b_rate': input_b_rate,
    }
    if INPUT_MODE == 'triple':
        new_row['input_c_rate'] = input_c_rate
    # Add all output neuron rates
    new_row.update(output_rates)
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


def already_tested(df, input_a_rate, input_b_rate, input_c_rate=0.0):
    """Check if this combination has already been tested."""
    if INPUT_MODE == 'triple':
        return ((df['input_a_rate'] == input_a_rate) & 
                (df['input_b_rate'] == input_b_rate) &
                (df['input_c_rate'] == input_c_rate)).any()
    else:
        return ((df['input_a_rate'] == input_a_rate) & 
                (df['input_b_rate'] == input_b_rate)).any()


def create_visualization(df):
    """Create visualization of the pathway response."""
    print("\nCreating visualization...")
    
    if INPUT_MODE == 'single':
        # Single input mode - create line plots
        create_single_input_visualization(df)
    elif INPUT_MODE == 'triple':
        # Triple input mode - create 3D slices
        create_triple_input_visualization(df)
    else:
        # Dual input mode - create heatmaps
        create_dual_input_visualization(df)


def create_single_input_visualization(df):
    """Create line plots for single input mode."""
    # Sort by input rate
    df_sorted = df.sort_values('input_a_rate')
    
    # Get list of output neurons
    output_names = list(OUTPUT_NEURONS.keys())
    n_outputs = len(output_names)
    
    # Create grid layout - 2 columns, enough rows
    n_cols = 2
    n_rows = (n_outputs + 1) // 2
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, neuron_name in enumerate(output_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        color = colors[idx % len(colors)]
        ax.plot(df_sorted['input_a_rate'], df_sorted[neuron_name], 'o-', 
                linewidth=2, markersize=6, color=color)
        ax.set_xlabel('Input Stimulation (Hz)')
        ax.set_ylabel(f'{neuron_name} Firing Rate (Hz)')
        ax.set_title(f'{neuron_name} Response')
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(n_outputs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"Saved single input visualization to {OUTPUT_PLOT}")
    plt.show()


def create_triple_input_visualization(df):
    """Create 3D visualization as 2D heatmap slices for triple input mode."""
    print("Creating 3D visualizations (2D slices)...")
    
    output_names = list(OUTPUT_NEURONS.keys())
    
    # Get unique values for input A (JO neurons) - we'll create slices at these JO rates
    jo_rates = sorted(df['input_a_rate'].unique())
    
    # Select subset of JO rates to visualize (e.g., min, 25%, 50%, 75%, max)
    if len(jo_rates) > 5:
        indices = [0, len(jo_rates)//4, len(jo_rates)//2, 3*len(jo_rates)//4, -1]
        jo_rates_to_plot = [jo_rates[i] for i in indices]
    else:
        jo_rates_to_plot = jo_rates
    
    # For each output neuron, create a figure with multiple heatmaps
    for neuron_name in output_names:
        n_slices = len(jo_rates_to_plot)
        n_cols = min(3, n_slices)  # Max 3 columns
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, jo_rate in enumerate(jo_rates_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Filter data for this JO rate
            df_slice = df[df['input_a_rate'] == jo_rate]
            
            # Create pivot table: LEFT_GRNs (input_b) vs RIGHT_GRNs (input_c)
            if len(df_slice) > 0:
                matrix = df_slice.pivot(index='input_c_rate', columns='input_b_rate', values=neuron_name)
                sns.heatmap(matrix, ax=ax, cmap='viridis',
                           cbar_kws={'label': f'{neuron_name} Firing Rate (Hz)'})
                ax.set_title(f'JO Rate = {jo_rate:.1f} Hz')
                ax.set_xlabel('Left GRN Stimulation (Hz)')
                ax.set_ylabel('Right GRN Stimulation (Hz)')
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'JO Rate = {jo_rate:.1f} Hz')
        
        # Hide unused subplots
        for idx in range(n_slices, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'{neuron_name} Response: Left vs Right GRN at Different JO Rates', fontsize=14, y=1.00)
        plt.tight_layout()
        
        # Save with neuron-specific filename
        plot_path = OUTPUT_PLOT.replace('.png', f'_{neuron_name}_3D.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D visualization for {neuron_name} to {plot_path}")
        plt.close()
    
    print("3D visualizations complete!")


def create_dual_input_visualization(df):
    """Create heatmap visualizations for dual input mode."""
    output_names = list(OUTPUT_NEURONS.keys())
    n_outputs = len(output_names)
    
    # Create grid layout for all output neurons
    n_cols = 2
    n_rows = (n_outputs + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create heatmap for each output neuron
    for idx, neuron_name in enumerate(output_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        matrix = df.pivot(index='input_b_rate', columns='input_a_rate', values=neuron_name)
        sns.heatmap(matrix, ax=ax, cmap='viridis',
                   cbar_kws={'label': f'{neuron_name} Firing Rate (Hz)'})
        ax.set_title(f'{neuron_name} Response')
        ax.set_xlabel('Input A Stimulation (Hz)')
        ax.set_ylabel('Input B Stimulation (Hz)')
        ax.invert_yaxis()
    
    # Hide any unused subplots
    for idx in range(n_outputs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"Saved dual input visualization to {OUTPUT_PLOT}")
    plt.show()
    
    # Create differential plots for left-right pairs
    create_differential_plots(df)


def create_differential_plots(df):
    """Create differential (L - R) heatmaps for paired output neurons."""
    # Find left-right pairs
    left_neurons = [name for name in OUTPUT_NEURONS.keys() if name.endswith('_L')]
    
    if not left_neurons:
        return
    
    n_pairs = len(left_neurons)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, left_name in enumerate(left_neurons):
        # Find corresponding right neuron
        right_name = left_name[:-2] + '_R'
        
        if right_name not in OUTPUT_NEURONS:
            continue
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Create differential matrix
        left_matrix = df.pivot(index='input_b_rate', columns='input_a_rate', values=left_name)
        right_matrix = df.pivot(index='input_b_rate', columns='input_a_rate', values=right_name)
        diff_matrix = left_matrix - right_matrix
        
        neuron_type = left_name[:-2]  # Remove '_L' suffix
        sns.heatmap(diff_matrix, ax=ax, cmap='RdBu_r', center=0,
                   cbar_kws={'label': f'{neuron_type} Differential (L - R) Hz'})
        ax.set_title(f'{neuron_type} Differential Response\n(Left - Right)')
        ax.set_xlabel('Input A Stimulation (Hz)')
        ax.set_ylabel('Input B Stimulation (Hz)')
        ax.invert_yaxis()
    
    # Hide any unused subplots
    for idx in range(n_pairs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    diff_plot_path = OUTPUT_PLOT.replace('.png', '_differential.png')
    plt.savefig(diff_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved differential visualization to {diff_plot_path}")
    plt.show()


def main():
    """Main execution function."""
    # Load connectome
    model, flyid2i, i2flyid, num_neurons = load_connectome()
    
    # Load existing results
    df = load_existing_results()
    
    # Generate all combinations to test based on mode
    if INPUT_MODE == 'single':
        total_combinations = len(RATES_A)
        combinations_to_test = []
        for rate in RATES_A:
            if not already_tested(df, rate, 0.0):
                combinations_to_test.append((rate, 0.0, 0.0))
    elif INPUT_MODE == 'triple':
        total_combinations = len(RATES_A) * len(RATES_B) * len(RATES_C)
        combinations_to_test = []
        for rate_a in RATES_A:  # JO neurons
            for rate_b in RATES_B:  # Left GRNs
                for rate_c in RATES_C:  # Right GRNs
                    if not already_tested(df, rate_a, rate_b, rate_c):
                        combinations_to_test.append((rate_a, rate_b, rate_c))
    else:  # dual mode
        total_combinations = len(RATES_A) * len(RATES_B)
        combinations_to_test = []
        for rate_a in RATES_A:
            for rate_b in RATES_B:
                if not already_tested(df, rate_a, rate_b):
                    combinations_to_test.append((rate_a, rate_b, 0.0))
    
    print(f"\nCombinations already tested: {len(df)}")
    print(f"Combinations remaining: {len(combinations_to_test)}")
    
    if len(combinations_to_test) == 0:
        print("\nAll combinations already tested!")
    else:
        print(f"\nStarting simulation sweep...")
        
        # Run simulations with progress bar
        if INPUT_MODE == 'single':
            desc = "Testing rates"
        elif INPUT_MODE == 'triple':
            desc = "Testing 3D combinations"
        else:
            desc = "Testing 2D combinations"
        
        for input_a_rate, input_b_rate, input_c_rate in tqdm(combinations_to_test, desc=desc):
            output_rates = run_simulation(
                model, flyid2i, input_a_rate, input_b_rate, input_c_rate
            )
            
            # Save result incrementally
            df = save_result(df, input_a_rate, input_b_rate, output_rates, input_c_rate)
    
    print(f"\nSimulation sweep complete!")
    print(f"Total results collected: {len(df)}")
    
    # Create visualizations
    if len(df) == total_combinations:
        create_visualization(df)
    else:
        print(f"\nWarning: Only {len(df)}/{total_combinations} combinations tested.")
        print(f"Run the script again to complete the sweep before visualizing.")


if __name__ == '__main__':
    main()
