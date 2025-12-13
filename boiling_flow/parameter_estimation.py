import numpy as np
import scipy
import boiling_flow._utils as utils
from boiling_flow.boiling_flow import generate_random_phase_screen
import time
from datetime import timedelta
import numbers

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

def estimate_outer_scale(aperture_shape, delta):
    """Set L0 to the length of the aperture [m].

    Args:
        aperture_shape (tuple): shape of the aperture [pixels]
        delta (float): grid sampling [m]

    Returns:
        **L0_est** (*float*) -- the estimated L0 [m]
    """
    assert (isinstance(aperture_shape, tuple) and (len(aperture_shape) == 2))
    assert (delta > 0)

    N = max(aperture_shape)
    L0_est = N * delta
    return L0_est


def estimate_spatial_psd(input_data, sampling_frequency):
    """Estimate the spatial PSD of the input data using Welch's method.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        sampling_frequency (float): sampling frequency of the input data

    Returns:
        **psd_estimate** (*ndarray*) -- numpy 2-D array of shape (N, N) containing the spatial PSD estimate
    """
    assert ((input_data.ndim == 3) and (input_data.shape[1] == input_data.shape[2]))
    assert ((input_data.shape[1] % 2) == 0)
    assert (sampling_frequency > 0)

    N = input_data.shape[1]

    # Creates a 2-D Hamming window to apply to the data:
    hamming_window = np.outer(np.hamming(N), np.hamming(N))[np.newaxis, :]
    welch_scaling_factor = np.sum(hamming_window ** 2)

    # Removes TTP from the data
    data_ttp_removed = utils.remove_ttp(input_data)

    # Applies the 2-D Hamming window to the TTP-removed data:
    windowed_data = hamming_window * data_ttp_removed

    # Calculates the 2-D FFT of the data:
    block_dft = utils.ft2(windowed_data)

    # Calculates the energy spectrum of the data:
    data_energy_spectrum = np.abs(block_dft) ** 2

    # Estimates the PSD of the data:
    psd_estimates = data_energy_spectrum / welch_scaling_factor

    # Averages over time and converts units from [energy per pixel^2] to [energy per m^2]
    psd_estimate = np.average(psd_estimates, axis=0) / (sampling_frequency ** 2)

    return psd_estimate


def calculate_von_karman_psd_terms(fx, fy, L0, gamma0):
    """Calculates the Von Karman PSD terms which exclude r0.

    Args:
        fx (ndarray): numpy array containing the frequency grid values with respect to the x-axis
        fy (ndarray): numpy array containing the frequency grid values with respect to the y-axis
        L0 (float): outer scale [m]
        gamma0 (float): anisotropy parameter

    Returns:
        **von_karman_psd** (*ndarray*) -- numpy 2-D array of shape (N, N) containing the anisotropic Von Karman PSD terms
    """
    assert (L0 > 0)

    von_karman_psd_terms = 0.023 * np.power(fx ** 2 + (gamma0 * (fy ** 2)) + np.power(L0, -2), -11 / 6)

    return von_karman_psd_terms


def fit_r0_from_spatial_psd(spatial_psd_estimate, fx, fy, L0, gamma0=1.0):
    """Calculate the best-fit r0 to the spatial PSD of the measured data.

    Args:
        spatial_psd_estimate (ndarray): numpy 2-D array of shape (N, N) containing the spatial PSD estimate of the
            measured data
        fx (ndarray): numpy array containing the frequency grid values with respect to the x-axis
        fy (ndarray): numpy array containing the frequency grid values with respect to the y-axis
        L0 (float): outer scale [m]
        gamma0 (float, optional): [Default=1.0] anisotropy parameter

    Returns:
        **r0_est** (*float*) -- the best-fit of r0 to the spatial PSD of the measured data
    """
    assert (L0 > 0)

    von_karman_psd_terms = calculate_von_karman_psd_terms(fx, fy, L0, gamma0)

    r0_vals = np.power(np.divide(von_karman_psd_terms, spatial_psd_estimate), 3/5)

    r0_est = np.average(r0_vals)

    return r0_est


def estimate_r0(input_data, delta, L0, frequency_mask=None):
    """Estimate r0 to fit the isotropic Von Karman PSD (i.e., with gamma0=1) to the spatial PSD of the measured data.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        delta (float): grid sampling [m]
        L0 (float): outer scale [m]
        frequency_mask (ndarray, optional): [Default=None] numpy boolean 2-D array indicating which frequencies to
            average over

            - If set to None, all frequencies except the center (zero) frequency bins are used.

    Returns:
        **r0_est** (*float*) -- the best-fit of r0 to the spatial PSD of the measured data
    """
    N = input_data.shape[1]
    sampling_frequency = 1 / delta  # [samples per m]

    frequency_bins = np.arange(-N/2, N/2) * 1.0 / (N*delta)
    fx, fy = np.meshgrid(frequency_bins, frequency_bins)    # frequency grid

    # Compute spatial PSD and structure function of the measured data
    spatial_psd_estimate = estimate_spatial_psd(input_data, sampling_frequency)

    # If a frequency mask is not provided, then remove the center frequencies (where either fx or fy is zero) to
    # issues caused by TTP-removal
    if frequency_mask is None:
        frequency_mask = np.full(shape=(N, N), fill_value=True, dtype=bool)
        frequency_mask[N // 2, :] = False
        frequency_mask[:, N // 2] = False

    # Fit r0 to the spatial PSD of the measured data
    r0_est = fit_r0_from_spatial_psd(spatial_psd_estimate, fx, fy, L0)

    return r0_est


def estimate_r0_and_gamma0(input_data, delta, L0, frequency_mask=None):
    """Estimate r0 and gamma0 to fit the anisotropic Von Karman PSD and structure function to the measured data's
    spatial PSD and structure function (respectively).

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        delta (float): grid sampling [m]
        L0 (float): outer scale [m]
        frequency_mask (ndarray, optional): [Default=None] numpy boolean 2-D array indicating which frequencies to
            average over

            - If set to None, all frequencies except the center (zero) frequency bins are used.

    Returns:
        **r0_est** (*float*) -- the estimate of r0
        **gamma0_est** (*float*) -- the estimate of gamma0
    """
    N = input_data.shape[1]
    sampling_frequency = 1 / delta  # [samples per m]

    frequency_bins = np.arange(-N/2, N/2) * 1.0 / (N*delta)
    fx, fy = np.meshgrid(frequency_bins, frequency_bins)    # frequency grid

    # Compute spatial PSD and structure function of the measured data
    spatial_psd_estimate = estimate_spatial_psd(input_data, sampling_frequency)
    gt_structure_function = utils.structure_function_2d(input_data, compute_square_root=True)[1]

    # If a frequency mask is not provided, then remove the center frequencies (where either fx or fy is zero) to
    # issues caused by TTP-removal
    if frequency_mask is None:
        frequency_mask = np.full(shape=(N, N), fill_value=True, dtype=bool)
        frequency_mask[N // 2, :] = False
        frequency_mask[:, N // 2] = False

    # Apply frequency mask to frequency bins and spatial PSD values
    fx, fy = fx[frequency_mask], fy[frequency_mask]
    spatial_psd_estimate = spatial_psd_estimate[frequency_mask]

    # Test 100 evenly-spaced values of gamma0 ranging from 0.1 to 2.0
    gamma0_vals = np.linspace(0.1, 2.0, 100)

    # r0 estimates and structure function errors on each iteration
    r0_ests = np.zeros(100)
    structure_function_error = np.zeros(100)

    # Test each gamma0
    for idx, gamma0 in enumerate(gamma0_vals):
        # Fit r0 to the spatial PSD of the measured data
        r0_est = fit_r0_from_spatial_psd(spatial_psd_estimate, fx, fy, L0, gamma0)
        r0_ests[idx] = r0_est

        # Generate random phase screens with parameters r0_est and gamma0 to compute the structure function of phase
        # screens with the current gamma0 and r0 values
        simulated_phase_screens = np.zeros(shape=(10_000, N, N))
        for screen_idx in range(10_000):
            random_oversize_screen = generate_random_phase_screen(4 * N, delta, L0, r0_est, gamma0)
            random_screen = random_oversize_screen[:N, :N]
            simulated_phase_screens[screen_idx] = utils.remove_ttp(random_screen)
        simulated_structure_function = utils.structure_function_2d(simulated_phase_screens, compute_square_root=True)[1]

        # Compute structure function error
        structure_function_error[idx] = np.average((simulated_structure_function - gt_structure_function) ** 2)

    # Choose gamma0 with the lowest structure function error
    min_error_idx = np.argmin(structure_function_error)
    gamma0_est = gamma0_vals[min_error_idx]

    # Use parabolic interpolation to refine the estimates
    if gamma0_est not in (gamma0_vals[0], gamma0_vals[-1]):
        gamma0_est = utils.parabolic_interpolation_1d(gamma0_vals[min_error_idx - 1:min_error_idx + 2],
                                                      structure_function_error[min_error_idx - 1:min_error_idx + 2])[0]

        r0_est = fit_r0_from_spatial_psd(spatial_psd_estimate, fx, fy, L0, gamma0_est)
    else:
        r0_est = r0_ests[min_error_idx]

    return r0_est, gamma0_est


def estimate_spatial_cross_correlation(input_data, time_lag):
    """Estimate the spatial cross-correlation of the input data with some time-lag.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        time_lag (int): the time-lag to use for the cross-correlation

    Returns:
        **spatial_cross_correlation** (*ndarray*) -- numpy 2-D array of shape (2N-1, 2N-1) containing the spatial
        cross-correlation estimate
    """
    assert ((input_data.ndim == 3) and (input_data.shape[1] == input_data.shape[2]))
    assert ((input_data.shape[1] % 2) == 0)
    assert (isinstance(time_lag, numbers.Integral) and (time_lag > 0))

    num_time_steps = input_data.shape[0]
    N = input_data.shape[1]

    spatial_cross_correlation = np.zeros((2 * N - 1, 2 * N - 1))

    for time_step_idx in range(time_lag, num_time_steps):
        spatial_cross_correlation += scipy.signal.correlate2d(input_data[time_step_idx],
                                                              input_data[time_step_idx - time_lag], mode='full')

    spatial_cross_correlation /= (num_time_steps - time_lag)

    # Average over the number of pairs for each bin
    ones_array = np.ones(input_data.shape[1:])
    num_pairs = scipy.signal.fftconvolve(ones_array, ones_array, mode='full')   # number of pixel pairs for each input
    spatial_cross_correlation /= num_pairs

    return spatial_cross_correlation


def maximize_cross_correlation(input_data, time_lag):
    """Find the location of the maximum of the spatial cross-correlation estimate. Refine the estimate using parabolic
    interpolation. Estimate the flow velocity components by dividing by the time-lag.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        time_lag (int): the time-lag to use for the cross-correlation

    Returns:
        - **v_x_est** (*float*) -- the flow velocity estimate with respect to the x-axis
        - **v_y_est** (*float*) -- the flow velocity estimate with respect to the y-axis
        - **max_correlation** (*float*) -- the maximum cross-correlation
    """
    assert ((input_data.ndim == 3) and (input_data.shape[1] == input_data.shape[2]))
    assert ((input_data.shape[1] % 2) == 0)
    assert (isinstance(time_lag, numbers.Integral) and (time_lag > 0))

    N = input_data.shape[1]

    spatial_cross_correlation = estimate_spatial_cross_correlation(input_data, time_lag)

    # Find maximal index of the spatial cross-correlation array
    y, x = np.unravel_index(np.argmax(spatial_cross_correlation), spatial_cross_correlation.shape)

    # Inputs to the cross-correlation array
    input_shifts_x = np.arange(x - 1, x + 2)
    input_shifts_y = np.arange(y - 1, y + 2)

    # Pixel shifts to use for interpolation
    x_values_interp = input_shifts_x - (N - 1)
    y_values_interp = input_shifts_y - (N - 1)

    # Use 2-D parabolic interpolation if (y, x) is not on the edge of the cross-correlation array
    if ((input_shifts_x[0] >= 0) and (x_values_interp[-1] < N) and (input_shifts_y[0] >= 0) and
            (y_values_interp[-1] < N)):
        input_shifts_X, input_shifts_Y = np.meshgrid(input_shifts_x, input_shifts_y)
        correlation_vals = spatial_cross_correlation[input_shifts_Y, input_shifts_X]
        x_max, y_max, max_correlation = utils.parabolic_interpolation_2d(x_values_interp, y_values_interp, correlation_vals)

    # Use 1-D parabolic interpolation if (y, x) is on the edge of the cross-correlation array
    elif (((input_shifts_x[0] < 0) or (x_values_interp[-1] >= N)) and (input_shifts_y[0] >= 0) and
          (y_values_interp[-1] < N)):
        # Interpolate along the y-axis
        x_max = x_values_interp[1]
        correlation_vals_y = spatial_cross_correlation[input_shifts_y, x]
        y_max, max_correlation = utils.parabolic_interpolation_1d(y_values_interp, correlation_vals_y)
    elif (((input_shifts_y[0] < 0) or (y_values_interp[-1] >= N)) and (input_shifts_x[0] >= 0) and
          (x_values_interp[-1] < N)):
        # Interpolate along the x-axis
        y_max = y_values_interp[1]
        correlation_vals_x = spatial_cross_correlation[y, input_shifts_x]
        x_max, max_correlation = utils.parabolic_interpolation_1d(x_values_interp, correlation_vals_x)

    # Do not interpolate if (y, x) is on the corner of the cross-correlation array
    else:
        x_max, y_max = x_values_interp[1], y_values_interp[1]
        max_correlation = spatial_cross_correlation[y, x]

    v_x_est, v_y_est = x_max / time_lag, y_max / time_lag

    return v_x_est, v_y_est, max_correlation


def estimate_flow_velocities(input_data, initial_time_lag=1, min_snr=10.0, num_sigma=5):
    """Estimate the flow velocity components by averaging over multiple time-lags.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        initial_time_lag (int, optional): [Default=1] the initial time lag to use for the cross-correlation estimate
        min_snr (float, optional): [Default=10.0] the minimum SNR to enforce for the max cross-correlation estimate
        num_sigma (int, optional): [Default=5] the standard deviation confidence to enforce for the max
            cross-correlation estimate

    Returns:
        - **v_x_est** (*float*) -- the flow velocity estimate with respect to the x-axis
        - **v_y_est** (*float*) -- the flow velocity estimate with respect to the y-axis
    """
    assert ((input_data.ndim == 3) and (input_data.shape[1] == input_data.shape[2]))
    assert ((input_data.shape[1] % 2) == 0)
    assert (isinstance(initial_time_lag, numbers.Integral) and (initial_time_lag > 0))
    assert (min_snr > 0)
    assert (isinstance(num_sigma, numbers.Integral) and (num_sigma > 0))

    num_time_steps = input_data.shape[0]
    N = input_data.shape[1]

    # Normalize the input data by removing the sample mean and standard deviation from each pixel
    mean_removed_data = input_data - np.mean(input_data, axis=0)
    normalized_data = mean_removed_data / np.sqrt(np.mean(mean_removed_data ** 2, axis=0))

    # Arrays containing each time-shift and (vx, vy) estimate
    time_shift_array = []
    v_x_array = []
    v_y_array = []

    # Initial estimate
    v_x, v_y = maximize_cross_correlation(input_data=normalized_data, time_lag=initial_time_lag)[:2]
    time_shift_array.append(initial_time_lag)
    v_x_array.append(v_x)
    v_y_array.append(v_y)

    # Find an initial value of the time-shift upper bound
    v_mag = np.sqrt(v_x**2+v_y**2)
    if np.divide(N-1, v_mag) == np.inf:
        time_lag_upper_bound = np.inf
    else:
        time_lag_upper_bound = int(np.floor((N-1)/v_mag))

    time_lag = initial_time_lag + 1

    while time_lag < time_lag_upper_bound:
        v_x, v_y, max_correlation = maximize_cross_correlation(input_data=normalized_data, time_lag=time_lag)

        # Enforce that the max correlation estimate has the desired SNR, with the desired confidence:
        if max_correlation < (min_snr + num_sigma) * np.sqrt(2) / np.sqrt(num_time_steps - time_lag):
            break

        # Upper the time-lag upper bound
        v_mag = np.sqrt(v_x**2+v_y**2)
        if np.divide(N-1, v_mag) != np.inf:
            if int(np.floor((N-1)/v_mag)) < time_lag_upper_bound:
                time_lag_upper_bound = int(np.floor((N-1)/v_mag))

                # If the new upper bound is larger than the time-lag, remove entries which exceed the upper bound
                if time_lag >= time_lag_upper_bound:
                    time_shift_array = np.array(time_shift_array)
                    v_x_array = np.array(v_x_array)
                    v_y_array = np.array(v_y_array)
                    if (time_shift_array >= time_lag_upper_bound).all():
                        v_x_array = np.array([v_x_array[0]])
                        v_y_array = np.array([v_y_array[0]])
                    else:
                        # idx = np.argmax(time_shift_array[(time_shift_array < time_lag_upper_bound)])
                        v_x_array = v_x_array[time_shift_array < time_lag_upper_bound]
                        v_y_array = v_y_array[time_shift_array < time_lag_upper_bound]
                    break
        time_shift_array.append(time_lag)
        v_x_array.append(v_x)
        v_y_array.append(v_y)
        time_lag = time_lag + 1

    v_x_est = np.average(v_x_array)
    v_y_est = np.average(v_y_array)

    return v_x_est, v_y_est


def estimate_alpha(input_data, v_x, v_y, frequency_mask=None):
    """Estimate the flow correlation parameter alpha given the flow velocity estimates (vx, vy).

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, N, N) containing the input data
        v_x (float): flow velocity component with respect to the x-axis [pixels per time-step]
        v_y (float): flow velocity component with respect to the y-axis [pixels per time-step]
        frequency_mask (ndarray, optional): [Default=None] numpy boolean 2-D of shape (N, N) specifying which frequency
            bins to include in the least-squares problem

            - If set to None, the function includes all frequency bins.

    Returns:
        **alpha** (*float*) -- the estimated flow correlation parameter in the range [0, 1]
    """
    assert ((input_data.ndim == 3) and (input_data.shape[1] == input_data.shape[2]))
    assert ((input_data.shape[1] % 2) == 0)
    if frequency_mask is not None:
        assert (frequency_mask.shape == input_data.shape[1:])
        assert (frequency_mask.dtype == bool)

    num_time_steps = input_data.shape[0]
    N = input_data.shape[1]

    frequency_bins = np.arange(-N / 2, N / 2) * 1.0 / N     # units of [cycles per pixel]
    fx, fy = np.meshgrid(frequency_bins, frequency_bins)    # frequency grid
    phase_shift = np.exp(-1j * 2. * np.pi * (v_x * fx + v_y * fy))

    # Use Hamming window for FFT
    hamming_window = np.hamming(N)
    window = np.outer(hamming_window, hamming_window)

    # Arrays for least-squares problem
    observations = np.zeros((num_time_steps - 1, N, N), dtype=complex)
    ground_truth = np.zeros((num_time_steps - 1, N, N), dtype=complex)

    for time_idx in range(num_time_steps - 1):
        # Apply the phase shift to the current phase screen
        current_phase_screen_ft = utils.ft2(input_data[time_idx])
        current_phase_screen_shifted_ft = phase_shift * current_phase_screen_ft
        current_phase_screen_shifted = utils.ift2(current_phase_screen_shifted_ft).real
        current_phase_screen_shifted_ttp_rem = utils.remove_ttp(current_phase_screen_shifted)
        # Return to frequency domain
        current_windowed_phase_screen_shifted_ttp_rem = window * current_phase_screen_shifted_ttp_rem
        observations[time_idx] = utils.ft2(current_windowed_phase_screen_shifted_ttp_rem)

        # Take the FFT of the next phase screen
        next_phase_screen_ttp_rem = utils.remove_ttp(input_data[time_idx + 1])
        windowed_next_phase_screen_ttp_rem = window * next_phase_screen_ttp_rem
        ground_truth[time_idx] = utils.ft2(windowed_next_phase_screen_ttp_rem)

    if frequency_mask is None:
        frequency_mask = np.full(shape=(N, N), fill_value=False, dtype=bool)

    # Apply frequency mask to both arrays
    observations = observations[:, frequency_mask].flatten()
    ground_truth = ground_truth[:, frequency_mask].flatten()

    # Find the least-squares solution
    alpha_est = np.vdot(observations, ground_truth) / np.vdot(observations, observations)

    return np.clip(np.real(alpha_est), 0, 1)


def estimate_boiling_flow_parameters(training_data, delta, frequency_bin_cutoff=2, mask=None, mode='anisotropic'):
    """Estimate the boiling flow parameters (L0, r0, vx, vy, alpha) from training data. Adapted from
    :cite:`UtleyBoiling`.

    Args:
        training_data (ndarray): numpy 3-D array of shape (num_time_steps, M, N) containing the training data
        delta (float): grid sampling [m]
        frequency_bin_cutoff (int, optional): [Default=2] cut-off frequency bin index for frequency mask
        mask (ndarray, optional): [Default=None] numpy boolean 2-D of shape (M, N) specifying which pixels to use

            - If set to None, all pixels are used.

        mode (string, optional): [Default='anisotropic'] the decision to estimate parameters for either isotropic or
            anisotropic phase screens (if not 'anisotropic', should be 'isotropic').

    Returns:
        - **L0** (*float*) -- the outer scale estimate [m]
        - **r0** (*float*) -- the Fried coherence length estimate [m]
        - **gamma0** (*float*) -- the anisotropy parameter estimate
        - **v_x** (*float*) -- the flow velocity estimate with respect to the x-axis
        - **v_y** (*float*) -- the flow velocity estimate with respect to the y-axis
        - **alpha** (*float*) -- the flow correlation parameter estimate in the range [0, 1]
    """
    assert (training_data.ndim == 3)
    if mask is not None:
        assert (mask.shape == training_data.shape[1:])
        assert (mask.dtype == bool)
    else:
        mask = np.ones(training_data.shape[1:], dtype=bool)
    assert (mode in ['anisotropic', 'isotropic'])

    print("\nBoiling Flow Parameter Estimation")
    print("=================================")
    print(f"Number of Time Steps: {training_data.shape[0]}")
    print(f"Image Size:           ({training_data.shape[1]}, {training_data.shape[2]})")
    print(f"Grid Spacing [m]:     {delta}")
    start_time = time.time()

    aperture_shape = training_data.shape[1:]
    L0 = estimate_outer_scale(aperture_shape, delta)

    # Find largest even-length square that can be inscribed in the aperture:
    square_data = utils.extract_largest_square(training_data, mask)
    N = square_data.shape[1]

    print(f"Square Length:        {square_data.shape[1]}")
    print("=================================\n")

    if frequency_bin_cutoff is not None:
        assert isinstance(frequency_bin_cutoff, numbers.Integral)
        assert ((frequency_bin_cutoff > 0) and (frequency_bin_cutoff < N // 2))

        # Exclude frequency bins determined by the cut-off
        frequency_mask = np.full(shape=(N, N), fill_value=False, dtype=bool)
        frequency_mask[frequency_bin_cutoff:-frequency_bin_cutoff, frequency_bin_cutoff:-frequency_bin_cutoff] = True
    else:
        frequency_mask = np.full(shape=(N, N), fill_value=True, dtype=bool)
    frequency_mask[N // 2, :] = False  # exclude zero frequencies along the y-axis
    frequency_mask[:, N // 2] = False  # exclude zero frequencies along the x-axis

    if mode == 'isotropic':
        gamma0 = 1.0
        r0 = estimate_r0(square_data, delta, L0, frequency_mask)
    elif mode == 'anisotropic':
        r0, gamma0 = estimate_r0_and_gamma0(square_data, delta, L0, frequency_mask)

    v_x, v_y = estimate_flow_velocities(square_data)

    alpha = estimate_alpha(square_data, v_x, v_y, frequency_mask=frequency_mask)

    runtime_in_seconds = time.time() - start_time
    elapsed_time = str(timedelta(seconds=runtime_in_seconds))
    print("Boiling Flow Parameter Estimation Completed in {} (hr:min:sec)\n".format(elapsed_time))

    return L0, r0, gamma0, v_x, v_y, alpha