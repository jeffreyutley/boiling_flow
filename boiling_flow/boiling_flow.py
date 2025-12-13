import numpy as np
import boiling_flow._utils as utils
import time
from datetime import timedelta
import numbers

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

def generate_random_phase_screen(N, delta, L0, r0, gamma0=1.0):
    """Generate a random phase screen from the Von Karman distribution. Adapted from :cite:`Schmidt`.

    Args:
        N (int): screen dimension
        delta (float): grid sampling [m]
        L0 (float): outer scale [m]
        r0 (float): Fried coherence length [m]
        gamma0 (float, optional): [Default=1.0] anisotropy parameter

    Returns:
        **phase_screen** (*ndarray*) -- numpy 2-D array of shape (N, N) containing the random phase screen
    """
    assert (isinstance(N, numbers.Integral) and (N > 0) and (N % 2 == 0))
    assert ((delta > 0) and (r0 > 0) and (L0 > 0))

    frequency_grid_spacing = 1/(N*delta)
    frequency_bins = np.arange(-N/2,N/2)*frequency_grid_spacing
    fx, fy = np.meshgrid(frequency_bins, frequency_bins)    # frequency grid

    von_karman_psd = 0.023 * np.power(r0, -5/3) * np.power(fx ** 2 + gamma0 * fy ** 2 + np.power(L0, -2), -11 / 6)
    von_karman_psd[N//2,N//2] = 0   # set D.C. power to zero

    complex_white_noise = np.random.randn(N,N) + 1j*np.random.randn(N,N)
    phase_screen_fourier_transform = frequency_grid_spacing * np.sqrt(von_karman_psd) * complex_white_noise

    phase_screen = np.real(utils.ift2(phase_screen_fourier_transform, N ** 2))

    return phase_screen


def boiling_flow_single_time_step(input_phase_screen, delta, L0, r0, v_x, v_y, alpha, gamma0=1.0):
    """Apply a single time-step of the boiling flow algorithm. Adapted from :cite:`Srinath`.

    Args:
        input_phase_screen (ndarray): numpy 2-D array of shape (N, N) containing the previous phase screen
        delta (float): grid sampling [m]
        L0 (float): outer scale [m]
        r0 (float): Fried coherence length [m]
        v_x (float): flow velocity component with respect to the x-axis [pixels per time-step]
        v_y (float): flow velocity component with respect to the y-axis [pixels per time-step]
        alpha (float): flow correlation parameter in the range [0, 1]
        gamma0 (float, optional): [Default=1.0] anisotropy parameter

    Returns:
        **output_phase_screen** (*ndarray*) -- numpy 2-D array of shape (N, N) containing the next phase screen
    """
    assert (input_phase_screen.shape[0] == input_phase_screen.shape[1])
    assert (input_phase_screen.shape[0] % 2 == 0)
    assert ((delta > 0) and (r0 > 0) and (L0 > 0))
    assert ((alpha >= 0) and (alpha <= 1))

    N = input_phase_screen.shape[0]

    frequency_bins = np.arange(-N/2,N/2)*1.0/N  # units of cycles/pixel
    fx, fy = np.meshgrid(frequency_bins, frequency_bins)    # frequency grid
    phase_shift = np.exp(-1j * 2. * np.pi * (v_x * fx + v_y * fy))

    random_phase_screen = generate_random_phase_screen(N=N, delta=delta, L0=L0, r0=r0, gamma0=gamma0)
    random_phase_screen_fourier_transform = utils.ft2(random_phase_screen)
    previous_phase_screen_fourier_transform = utils.ft2(input_phase_screen)
    output_phase_screen_fourier_transform = (alpha * phase_shift * previous_phase_screen_fourier_transform +
                                           np.sqrt(1 - alpha ** 2) * random_phase_screen_fourier_transform)
    output_phase_screen = utils.ift2(output_phase_screen_fourier_transform).real

    return output_phase_screen


def generate_boiling_flow_data(num_time_steps, N, delta, L0, r0, v_x, v_y, alpha, gamma0=1.0, k=4, remove_ttp=True):
    """Generate a time-series of phase screens using the boiling flow algorithm.

    Args:
        num_time_steps (int): number of time steps to generate
        N (int): screen dimension
        delta (float): grid sampling [m]
        L0 (float): outer scale [m]
        r0 (float): Fried coherence length [m]
        v_x (float): flow velocity component with respect to the x-axis [pixels per time-step]
        v_y (float): flow velocity component with respect to the y-axis [pixels per time-step]
        alpha (float): flow correlation parameter in the range [0, 1]
        gamma0 (float, optional): [Default=1.0] anisotropy parameter
        k (int, optional): [Default=4] scale factor for over-sized phase screens
        remove_ttp (bool, optional): whether to remove tip, tilt, and piston (TTP) from the output phase screens

    Returns:
        **output_phase_screens** (*ndarray*) -- numpy 3-D array of shape (num_time_steps, N, N) containing time-series
        of phase screens
    """
    assert (isinstance(num_time_steps, numbers.Integral) and (num_time_steps > 0))
    assert (isinstance(N, numbers.Integral) and (N > 0) and (N % 2 == 0))
    assert ((delta > 0) and (r0 > 0) and (L0 > 0))
    assert ((alpha >= 0) and (alpha <= 1))
    assert (isinstance(k, numbers.Integral) and (k > 0))
    assert (isinstance(remove_ttp, bool))

    print(f"\nBoiling Flow Data Generation: {num_time_steps} time-steps")
    print("==============================================")
    start_time = time.time()

    N_oversize = k * N  # over-sized phase screen dimension
    output_phase_screens = np.zeros((num_time_steps, N, N))

    # Initial (oversized) phase screen
    first_oversize_phase_screen = generate_random_phase_screen(N=N_oversize, delta=delta, L0=L0, r0=r0, gamma0=gamma0)

    # First (true) phase screen
    first_phase_screen = first_oversize_phase_screen[:N, :N]
    output_phase_screens[0, :, :] = first_phase_screen

    previous_oversize_phase_screen = first_oversize_phase_screen
    for time_step in range(1, num_time_steps):
        # Run a single step of boiling flow
        next_oversize_phase_screen = boiling_flow_single_time_step(input_phase_screen=previous_oversize_phase_screen,
                                                                   delta=delta, L0=L0, r0=r0, v_x=v_x, v_y=v_y,
                                                                   alpha=alpha, gamma0=gamma0)

        # Next (true) phase screen
        next_phase_screen = next_oversize_phase_screen[:N, :N]
        output_phase_screens[time_step, :, :] = next_phase_screen

        previous_oversize_phase_screen = next_oversize_phase_screen

    # Removes TTP if prompted
    if remove_ttp:
        output_phase_screens = utils.remove_ttp(output_phase_screens)

    runtime_in_seconds = time.time() - start_time
    elapsed_time = str(timedelta(seconds=runtime_in_seconds))
    print("Boiling Flow Data Generation Completed in {} (hr:min:sec)\n".format(elapsed_time))

    return output_phase_screens
