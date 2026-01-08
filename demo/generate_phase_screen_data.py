import boiling_flow
import numpy as np
import demo_utils
import os

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

"""Generate a time-series phase screens using boiling flow and the parameter estimates from measured data"""

datasets = ['F06', 'F12']
phase_psd_block_sizes = {'F06': 596, 'F12': 994}
flow_psd_block_sizes = {'F06': 298, 'F12': 496}

for dataset in datasets:
    print("Data Set %s" % dataset + '\n'
          "============\n"
          "============\n")

    # Build filepath for the TBL data set .npz file
    data_filepath = f'./demo/data/TBL_data/TBL_data_set_{dataset}.npz'

    print(f"-- Loading TBL_data_set_{dataset}.npz ---")
    print(f"Looking for file at: {os.path.abspath(data_filepath)}")

    # Load .npz data file
    try:
        data_file = np.load(data_filepath)
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

    phase_screens = data_file['phase_screens']  # [radians]
    mask = data_file['mask']
    pixel_spacing = data_file['pixel_spacing']  # [m]
    sampling_frequency = data_file['sampling_frequency'] # [Hz]
    wavelength = data_file['wavelength'] # [m]

    print(f"--- Done loading TBL_data_set_{dataset}.npz ---\n")

    # Boiling flow parameters - estimated from measured data
    #
    # Build filepath for the parameter estimates .npy file
    parameter_filepath = f'./demo/output/TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy'

    print(f"-- Loading TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy ---")
    print(f"Looking for file at: {os.path.abspath(parameter_filepath)}")

    # Load .npy data file
    try:
        parameter_estimates = np.load(parameter_filepath)
        print("File loaded successfully.")
    except FileNotFoundError:
        print("❌ ERROR: File not found!")
        print(f"Check that the file demo/output/TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy exists.")
        print("If the file does not exist, run the demo script demo/parameter_estimates_from_measured_data.py")
        raise
    except Exception as e:
        print("❌ ERROR while loading the .npy file:")
        print(e)
        raise

    # Extract parameters
    print("Extracting parameters from file...")
    L0, r0, gamma0, v_x, v_y, alpha = parameter_estimates

    print(f"--- Done loading TBL_data_set_{dataset}_boiling_flow_parameter_estimates.npy ---")

    # Extract the "outside mask" and training and testing data from the TBL data
    outside_mask = (np.average(1 - np.uint8(np.isnan(phase_screens)), axis=0) != 1)   # values to mask out
    num_training_time_steps = int(0.8 * phase_screens.shape[0])     # use 80% of the time series for training
    num_synthetic_time_steps = phase_screens.shape[0] - num_training_time_steps # compare with last 20% of time series
    measured_data_phase = phase_screens[num_training_time_steps:]

    # Phase TPS, Flow TPS, RMS, and structure function of the measured data
    measured_data_opd = measured_data_phase * wavelength / (2 * np.pi)  # convert to optical path difference (OPD) [m]
    phase_psd_block_size = phase_psd_block_sizes[dataset]
    flow_psd_block_size = flow_psd_block_sizes[dataset]
    phase_frequencies, measured_phase_tps = demo_utils.temporal_psd(data_values=measured_data_phase,
                                                                    time_block_size=phase_psd_block_size,
                                                                    mask=mask,
                                                                    sampling_frequency=sampling_frequency)
    flow_frequencies, measured_flow_tps = demo_utils.slopes_psd(data_values=measured_data_opd,
                                                                locations=pixel_spacing,
                                                                time_block_size=flow_psd_block_size,
                                                                sampling_frequency=sampling_frequency)
    measured_opd_rms = demo_utils.compute_rms(data_values=measured_data_opd,
                                              mask=mask)
    structure_function_inputs, measured_structure_function = (
        boiling_flow.utils.structure_function_2d(data_values=measured_data_phase,
                                                 mask=mask,
                                                 compute_square_root=False))
    measured_structure_function_sqrt = boiling_flow.utils.structure_function_2d(data_values=measured_data_phase,
                                                                                mask=mask,
                                                                                compute_square_root=True)[1]

    # Set phase screen size to the largest of the two axes
    N = np.max(phase_screens.shape[1:])
    if N % 2 == 1:
        N += 1

    # Generate boiling flow data
    boiling_flow_data = boiling_flow.generate_boiling_flow_data(num_time_steps=num_synthetic_time_steps,
                                                                N=N,
                                                                delta=pixel_spacing,
                                                                L0=L0,
                                                                r0=r0,
                                                                v_x=v_x,
                                                                v_y=v_y,
                                                                alpha=alpha,
                                                                gamma0=gamma0)

    boiling_flow_data = boiling_flow_data[:, :phase_screens.shape[1], :phase_screens.shape[2]]
    boiling_flow_data[:, outside_mask] = np.nan     # mask out pixels
    boiling_flow_data_opd = boiling_flow_data * (wavelength / (2 * np.pi))    # convert to OPD [m]

    # Phase TPS, Flow TPS, RMS, and structure function of the boiling flow data
    boiling_flow_phase_tps = demo_utils.temporal_psd(data_values=boiling_flow_data,
                                                     time_block_size=phase_psd_block_size,
                                                     mask=mask,
                                                     sampling_frequency=sampling_frequency)[1]
    boiling_flow_flow_tps = demo_utils.slopes_psd(data_values=boiling_flow_data_opd,
                                                  locations=pixel_spacing,
                                                  time_block_size=flow_psd_block_size,
                                                  sampling_frequency=sampling_frequency)[1]
    boiling_flow_opd_rms = demo_utils.compute_rms(data_values=boiling_flow_data_opd,
                                                  mask=mask)
    boiling_flow_structure_function = boiling_flow.utils.structure_function_2d(data_values=boiling_flow_data,
                                                                               mask=mask,
                                                                               compute_square_root=False)[1]
    boiling_flow_structure_function_sqrt = boiling_flow.utils.structure_function_2d(data_values=boiling_flow_data,
                                                                                    mask=mask,
                                                                                    compute_square_root=True)[1]

    # Video of measured and synthetic phase screens
    video = demo_utils.create_video(data=measured_data_phase[:500], title='Measured Data', mask=mask,
                                    data2=boiling_flow_data[:500], title2='Boiling Flow')
    video.save(f'./demo/output/TBL_data_set_{dataset}_comparison_video.gif')

    # Plot the TPS and structure functions
    demo_utils.plot_tps(frequencies=phase_frequencies, tps_values=measured_phase_tps, xlabel='Frequency $f$ [Hz]',
                        ylabel='PSD $S_{\\phi}(f)$ [energy/s]', title=f'Data Set {dataset}: Phase TPS',
                        tps_values2=boiling_flow_phase_tps, label1='Measured Data', label2='Boiling Flow',
                        savefile=f'./demo/output/TBL_data_set_{dataset}_phase_tps.pdf')
    demo_utils.plot_tps(frequencies=flow_frequencies, tps_values=measured_flow_tps, xlabel='Frequency $f$ [Hz]',
                        ylabel='PSD $S_{\\theta_x}(f)$ [energy/s]', title=f'Data Set {dataset}: Flow TPS',
                        tps_values2=boiling_flow_flow_tps, label1='Measured Data', label2='Boiling Flow',
                        savefile=f'./demo/output/TBL_data_set_{dataset}_flow_tps.pdf')
    demo_utils.structure_function_image(structure_function_inputs, measured_structure_function, title='Measured Data',
                                        x_label='$x/\\Delta$', y_label='$y/\\Delta$',
                                        cbar_label='$D_{\\phi}(x/\\Delta, y/\\Delta)$',
                                        structure_function_values_2=boiling_flow_structure_function,
                                        title_2='Boiling Flow', suptitle='Structure Function',
                                        savefile=f'./demo/output/TBL_data_set_{dataset}_structure_function.pdf')

    # Scalar metrics
    phase_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_phase_tps,
                                               estimated_data=boiling_flow_phase_tps)
    flow_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_flow_tps,
                                              estimated_data=boiling_flow_flow_tps)
    structure_function_error = demo_utils.compute_nrmse(ground_truth_data=measured_structure_function_sqrt,
                                                        estimated_data=boiling_flow_structure_function_sqrt)
    opd_rms_error = np.abs(measured_opd_rms - boiling_flow_opd_rms) / measured_opd_rms

    print("\n================================================")
    print(f"     Measured Data Set {dataset}: Scalar Metric Values")
    print("================================================")

    print(f"{'Phase TPS Error:':35s} {phase_tps_error}")
    print(f"{'Flow TPS Error:':35s} {flow_tps_error}")
    print(f"{'Structure Function Error:':35s} {structure_function_error}")
    print(f"{'OPD_rms Error:':35s} {opd_rms_error}\n")
