import boiling_flow
import numpy as np
import os

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

"""Estimate the boiling flow parameters from TBL data sets F06 and F12"""

datasets = ['F06', 'F12']

for dataset in datasets:
    print("Data Set %s" % dataset + '\n'
          "============\n"
          "============\n")

    # Build filepath
    filepath = f'./demo/data/TBL_data/TBL_data_set_{dataset}.npz'

    print(f"-- Loading TBL_data_set_{dataset}.npz ---")
    print(f"Looking for file at: {os.path.abspath(filepath)}")

    # Load .npz data file
    try:
        file = np.load(filepath)
        print("File loaded successfully.")
    except FileNotFoundError:
        print("❌ ERROR: File not found!")
        print("Check that the TBL_data folder was downloaded and placed correctly.")
        raise
    except Exception as e:
        print("❌ ERROR while loading the .npz file:")
        print(e)
        raise

    # Extract arrays
    print("Extracting arrays from file...")

    phase_screens = file['phase_screens']  # [radians]
    mask = file['mask']
    pixel_spacing = file['pixel_spacing']  # [m]

    print(f"--- Done loading TBL_data_set_{dataset}.npz ---\n")

    num_training_time_steps = int(0.8 * phase_screens.shape[0]) # use 80% of the time series for training
    training_data = phase_screens[:num_training_time_steps]

    parameter_estimates = boiling_flow.estimate_boiling_flow_parameters(training_data=training_data,
                                                                        delta=pixel_spacing,
                                                                        frequency_bin_cutoff=2,
                                                                        mask=mask,
                                                                        mode='anisotropic')

    L0, r0, gamma0, vx, vy, alpha = parameter_estimates

    print("\n================================================")
    print(f"         Data Set {dataset}: Parameter Estimates")
    print("================================================")

    print(f"{'Outer Scale L0 [m]:':45s} {L0: .5f}")
    print(f"{'Fried Coherence Length r0 [m]:':45s} {r0: .5f}")
    print(f"{'Anisotropy Parameter gamma0:':45s} {gamma0: .5f}")
    print(f"{'Flow Velocity (vx, vy) [pixels/step]:':45s} ({vx: .5f}, {vy: .5f})")
    print(f"{'Flow Correlation Parameter alpha:':45s} {alpha: .5f}\n")

    np.save(f'./demo/output/TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy', parameter_estimates)