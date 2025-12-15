import boiling_flow
import numpy as np

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

"""Estimate the boiling flow parameters from TBL data sets F06 and F12"""

datasets = ['F06', 'F12']

for dataset in datasets:
    file = np.load(f'./demo/data/TBL_data_set_{dataset}.npz')
    phase_screens = file['phase_screens']   # [radians]
    mask = file['mask']
    pixel_spacing = file['pixel_spacing']   # [m]

    num_training_time_steps = int(0.8 * phase_screens.shape[0]) # use 80% of the time series for training
    training_data = phase_screens[:num_training_time_steps]

    parameter_estimates = boiling_flow.estimate_boiling_flow_parameters(training_data=training_data,
                                                                        delta=pixel_spacing,
                                                                        frequency_bin_cutoff=2,
                                                                        mask=mask,
                                                                        mode='anisotropic')

    L0, r0, gamma0, vx, vy, alpha = parameter_estimates

    print(f"Data Set {dataset}: Parameter Estimates")
    print("=================================")
    print(f"Outer Scale L0 [m]:                                       {L0:.5f}")
    print(f"Fried Coherence Length r0 [m]:                            {r0:.5f}")
    print(f"Anisotropy Parameter gamma0:                              {gamma0:.5f}")
    print(f"Flow Velocity Components (vx, vy) [pixels per time-step]: ({vx:.5f}, {vy:.5f})")
    print(f"Flow Correlation Parameter alpha:                         {alpha:.5f}\n")

    np.save(f'./demo/output/TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy', parameter_estimates)