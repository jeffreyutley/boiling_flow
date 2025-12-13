import numpy as np
import demo_utils
import boiling_flow

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

"""Show results for an example simulated data set"""

# Ground-truth parameters
N = 32                  # [pixels]
L0 = 0.044              # [m]
delta = L0 / N          # [m]
r0 = 0.0044             # [m]
gamma0 = 1.0            # unitless
v_x = 1.0               # [pixels per time-step]
v_y = 0.0               # [pixels per time-step]
alpha = 0.9             # unitless
wavelength = 5.32e-7    # [m]
f_s = 100_000           # [Hz]

# TPS block sizes
phase_psd_block_size = 298
flow_psd_block_size = 148

num_training_time_steps = 10_000
num_evaluation_time_steps = 50_000

# Generate training and error evaluation data
training_data = boiling_flow.generate_boiling_flow_data(num_time_steps=num_training_time_steps,
                                                        N=N,
                                                        delta=delta,
                                                        L0=L0,
                                                        r0=r0,
                                                        v_x=v_x,
                                                        v_y=v_y,
                                                        alpha=alpha,
                                                        gamma0=gamma0)
evaluation_data = boiling_flow.generate_boiling_flow_data(num_time_steps=num_evaluation_time_steps,
                                                          N=N,
                                                          delta=delta,
                                                          L0=L0,
                                                          r0=r0,
                                                          v_x=v_x,
                                                          v_y=v_y,
                                                          alpha=alpha,
                                                          gamma0=gamma0)

# Estimate parameters
parameter_estimates = boiling_flow.estimate_boiling_flow_parameters(training_data=training_data,
                                                                    delta=delta,
                                                                    frequency_bin_cutoff=2,
                                                                    mode='isotropic')
L0_est, r0_est, gamma0_est, v_x_est, v_y_est, alpha_est = parameter_estimates

# Generate data using parameter estimates
generated_data = boiling_flow.generate_boiling_flow_data(num_time_steps=num_evaluation_time_steps,
                                                         N=N,
                                                         delta=delta,
                                                         L0=L0_est,
                                                         r0=r0_est,
                                                         v_x=v_x_est,
                                                         v_y=v_y_est,
                                                         alpha=alpha_est,
                                                         gamma0=gamma0_est)

print(f"Simulated Data: Parameter Estimate Errors")
print("==========================================")
print("r0 Relative Error:            ", np.abs(r0_est - r0) / r0)
print("Flow Velocity Relative Error: ", np.sqrt((v_x_est - v_x) ** 2 + v_y_est**2) / v_x)
print("alpha Relative Error:         ", np.abs(alpha_est - alpha) / alpha, '\n')

# Phase TPS, Flow TPS, RMS, and structure function of the evaluation data
evaluation_data_opd = evaluation_data * wavelength / (2 * np.pi)  # convert to optical path difference (OPD) [m]
phase_frequencies, measured_phase_tps = demo_utils.temporal_psd(data_values=evaluation_data,
                                                                time_block_size=phase_psd_block_size,
                                                                sampling_frequency=f_s)
flow_frequencies, measured_flow_tps = demo_utils.slopes_psd(data_values=evaluation_data_opd,
                                                            locations=delta,
                                                            time_block_size=flow_psd_block_size,
                                                            sampling_frequency=f_s)
measured_opd_rms = demo_utils.compute_rms(data_values=evaluation_data_opd)
structure_function_inputs, measured_structure_function = (
    boiling_flow.utils.structure_function_2d(data_values=evaluation_data,
                                             compute_square_root=False))
measured_structure_function_sqrt = boiling_flow.utils.structure_function_2d(data_values=evaluation_data,
                                                                            compute_square_root=True)[1]

generated_data_opd = generated_data * (wavelength / (2 * np.pi))  # convert to OPD [m]

# Phase TPS, Flow TPS, RMS, and structure function of the boiling flow data
boiling_flow_phase_tps = demo_utils.temporal_psd(data_values=generated_data,
                                                 time_block_size=phase_psd_block_size,
                                                 sampling_frequency=f_s)[1]
boiling_flow_flow_tps = demo_utils.slopes_psd(data_values=generated_data_opd,
                                              locations=delta,
                                              time_block_size=flow_psd_block_size,
                                              sampling_frequency=f_s)[1]
boiling_flow_opd_rms = demo_utils.compute_rms(data_values=generated_data_opd)
boiling_flow_structure_function = boiling_flow.utils.structure_function_2d(data_values=generated_data,
                                                                           compute_square_root=False)[1]
boiling_flow_structure_function_sqrt = boiling_flow.utils.structure_function_2d(data_values=generated_data,
                                                                                compute_square_root=True)[1]

# Video of measured and synthetic phase screens
video = demo_utils.create_video(data=evaluation_data[:500], title='Input Data', data2=generated_data[:500],
                                title2='Generated Data')
video.save(f'./demo/output/simulated_data_comparison_video.gif')

# Plot the TPS and structure functions
demo_utils.plot_tps(frequencies=phase_frequencies, tps_values=measured_phase_tps, xlabel='Frequency $f$ [Hz]',
                    ylabel='PSD $S_{\\phi}(f)$ [energy/s]', title='Simulated Data: Phase TPS',
                    tps_values2=boiling_flow_phase_tps, label1='Input Data', label2='Generated Data',
                    savefile=f'./demo/output/simulated_data_phase_tps.pdf')
demo_utils.plot_tps(frequencies=flow_frequencies, tps_values=measured_flow_tps, xlabel='Frequency $f$ [Hz]',
                    ylabel='PSD $S_{\\theta_x}(f)$ [energy/s]', title='Simulated Data: Flow TPS',
                    tps_values2=boiling_flow_flow_tps, label1='Input Data', label2='Generated Data',
                    savefile=f'./demo/output/simulated_data_flow_tps.pdf')
demo_utils.structure_function_image(structure_function_inputs, measured_structure_function, title='Input Data',
                                    structure_function_values_2=boiling_flow_structure_function,
                                    title_2='Generated Data', suptitle='Simulated Data: Structure Function',
                                    savefile=f'./demo/output/simulated_data_structure_function.pdf')

# Scalar metrics
phase_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_phase_tps,
                                           estimated_data=boiling_flow_phase_tps)
flow_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_flow_tps,
                                          estimated_data=boiling_flow_flow_tps)
structure_function_error = demo_utils.compute_nrmse(ground_truth_data=measured_structure_function_sqrt,
                                                    estimated_data=boiling_flow_structure_function_sqrt)
opd_rms_error = np.abs(measured_opd_rms - boiling_flow_opd_rms) / measured_opd_rms

print(f"Simulated Data: Error Metric Values")
print("===========================================")
print("Phase TPS Error:          ", phase_tps_error)
print("Flow TPS Error:           ", flow_tps_error)
print("Structure Function Error: ", structure_function_error)
print("OPD_rms Error:            ", opd_rms_error, '\n')
