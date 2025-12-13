import numpy as np

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

def ft2(data, scale=1.0):
    """Use FFT-shifts to take the 2-D FFT of data. Adapted from :cite:`Schmidt`.

    Args:
        data (ndarray): numpy 2-D or 3-D array of shape (N, N) or (num_time_steps, N, N) containing the data to take the
            Fourier transform of
        scale (float, optional): [Default = 1] scale of the FFT

    Returns:
        **scaled_fourier_transform** (*ndarray*) -- numpy 2-D or 3-D array of shape (N, N) or (num_time_steps, N, N)
        containing the FFT of the data
    """
    assert ((data.ndim in [2, 3]) and (data.shape[-2] == data.shape[-1]))
    if data.ndim == 3:
        fourier_transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
    else:
        fourier_transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data)))

    scaled_fourier_transform = fourier_transform * scale

    return scaled_fourier_transform


def ift2(data, scale=1.0):
    """Use FFT-shifts to take the inverse 2-D FFT of data. Adapted from :cite:`Schmidt`.

    Args:
        data (ndarray): numpy 2-D or 3-D array of shape (N, N) or (num_time_steps, N, N) containing the data to take the
            inverse Fourier transform of
        scale (float, optional): [Default = 1] scale of the inverse FFT

    Returns:
        **scaled_inverse_fourier_transform** (*ndarray*) -- numpy 2-D or 3-D array of shape (N, N) or
        (num_time_steps, N, N) containing the inverse FFT of the data
    """
    assert ((data.ndim in [2, 3]) and (data.shape[-2] == data.shape[-1]))
    if data.ndim == 3:
        inverse_fourier_transform = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(data, axes=(1, 2)), axes=(1, 2)),
                                                     axes=(1, 2))
    else:
        inverse_fourier_transform = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(data)))

    scaled_inverse_fourier_transform = inverse_fourier_transform * scale

    return scaled_inverse_fourier_transform


def remove_ttp(input_data):
    """
    Remove the tip, tilt, and piston (TTP) components from a phase screen or time-series of phase screens.

    Args:
        input_data (ndarray, float): numpy 2-D or 3-D array of shape (N, N) or (num_time_steps, N, N) containing the
            phase screen(s)

    Returns:
        **phase_screen_ttp_removed** (*ndarray*) -- numpy 2-D or 3-D array of shape (N, N) or (num_time_steps, N, N)
        containing the TTP-removed phase screen(s)
    """
    assert (input_data.ndim in [2, 3]) and (input_data.shape[-2] == input_data.shape[-1])
    N = input_data.shape[-1]

    if input_data.ndim == 2:
        phase_screens = input_data[np.newaxis, :, :].copy()
    else:
        phase_screens = input_data.copy()

    phase_screen_ttp_removed = np.zeros_like(phase_screens)

    for time_index in range(phase_screens.shape[0]):
        # Remove initial mean
        phase_screen_mean_rem = phase_screens[time_index] - phase_screens[time_index].mean()

        y, x = np.mgrid[0:N, 0:N]  # grid of (x, y) locations

        # Flatten arrays for linear regression
        observations = np.column_stack((x.ravel(), y.ravel(), np.ones(N*N)))   # basis of 2-D linear function
        ground_truth = phase_screen_mean_rem.ravel()    # phase screen values

        # Fit a plane to the phase screen data
        linear_coefficients = np.linalg.lstsq(observations, ground_truth, rcond=None)[0]
        a, b, c = linear_coefficients
        least_squares_plane = (a * x + b * y + c)

        # Subtract the plane from the original array to remove tip and tilt
        phase_screen_tt_rem = phase_screen_mean_rem - least_squares_plane

        # Subtract the spatial mean again to remove piston
        phase_screen_ttp_removed[time_index] = phase_screen_tt_rem - phase_screen_tt_rem.mean()

    phase_screen_ttp_removed = np.squeeze(phase_screen_ttp_removed)

    return phase_screen_ttp_removed



def parabolic_interpolation_2d(x_vals, y_vals, function_values):
    """Fit a 2-D parabola to a 3x3 grid of values centered at the maximum of a discrete 2-D array.

    Args:
        x_vals (ndarray): numpy 1-D array of shape (3,) containing x-values centered at maximal x-index
        y_vals (ndarray): numpy 1-D array of shape (3,) containing y-values centered at maximal y-index
        function_values (ndarray): numpy 2-D array of shape (3, 3) containing the function values centered at the
            maximum value.

    Returns:
        - **x_max** (*float*) -- interpolated x-coordinate of the maximum value
        - **y_max** (*float*) -- interpolated y-coordinate of the maximum value
        - **max_val** (*float*) -- interpolated maximum value
    """
    # Meshgrid: shapes (3, 3)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Flattened arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    f_flat = function_values.flatten()

    # Matrix of values to compute regression over
    Z = np.column_stack([X_flat**2, Y_flat**2, X_flat*Y_flat, X_flat, Y_flat, np.ones_like(X_flat)])

    # Solve for the coefficients
    parabola_coefficients = np.linalg.lstsq(Z, f_flat, rcond=None)[0]
    a, b, c, d, e, f = parabola_coefficients

    # Solve directly for the maximal index
    A = np.array([[2*a, c], [c, 2*b]])
    B = np.array([-d, -e])
    x_max, y_max = np.linalg.solve(A, B)

    # Plug in maximum to get the maximum value
    max_val = a * x_max ** 2 + b * y_max ** 2 + c * x_max * y_max + d * x_max + e * y_max + f

    return x_max, y_max, max_val


def parabolic_interpolation_1d(input_vals, function_values):
    """Fit a 1-D parabola to a three values centered at the maximum of a discrete array.

    Args:
        input_vals (ndarray): numpy 1-D array of shape (3,) containing the function inputs, centered at maximal index
        function_values (ndarray): numpy 1-D array of shape (3,) containing the function values centered at the
            maximum value.

    Returns:
        - **maximal_input** (*float*) -- interpolated input at the maximum value
        - **max_val** (*float*) -- interpolated maximum value
    """
    # Solve for x_max
    polynomial_coefficients = np.polyfit(input_vals, function_values, 2)
    a, b, c = polynomial_coefficients
    maximal_input = -b / (2 * a)
    max_val = a * maximal_input ** 2 + b * maximal_input + c

    return maximal_input, max_val


def extract_largest_square(input_data, mask=None):
    """
    Extracts the largest even-length square sub-array from an input 3-D array according to a mask.

    Args:
        input_data (ndarray): numpy 3-D array of shape (num_time_steps, M, N) containing the input data values
        mask (ndarray, optional): [Default=None] numpy boolean 2-D array of shape (M, N) indicating which values at
            each time-step of 'input_data' to allow in the square.

            - If set to None, all values are allowed.

    Returns:
        **square_data** (*ndarray*): numpy 3-D array of shape (num_time_steps, K, K) containing the largest even-length
        square (length K) that can be extracted from an input 3-D array.
    """
    if mask is not None:
        assert (mask.shape == input_data.shape[1:])
        assert (mask.dtype == bool)
    else:
        mask = np.ones(input_data.shape[1:], dtype=bool)

    M, N = mask.shape

    # Array used to hold the largest square sizes along the previous column
    largest_square_sizes_prev = np.zeros(N + 1, dtype=int)
    max_square_length = 0         # Length of the largest square found
    largest_square_endpoint = (0, 0)     # Bottom-right corner of the largest square (1-based indexing)

    for i in range(M):
        # Array used to hold the largest square sizes along the current column (index i)
        largest_square_sizes_current = np.zeros(N + 1, dtype=int)
        for j in range(1, N + 1):
            if mask[i, j - 1]:
                # Compute the size of the largest square ending at (i, j-1)
                largest_square_sizes_current[j] = min(largest_square_sizes_prev[j], largest_square_sizes_current[j - 1],
                                                      largest_square_sizes_prev[j - 1]) + 1
                # Update the maximum square length and endpoint if necessary
                if largest_square_sizes_current[j] > max_square_length:
                    max_square_length = largest_square_sizes_current[j]
                    largest_square_endpoint = (i + 1, j)  # Store 1-based index for easier slicing
        largest_square_sizes_prev = largest_square_sizes_current  # Move current row to previous for next iteration

    # If no square can be inscribed, return an empty array
    if max_square_length == 0:
        return np.array([])

    # Extract the square from the original data
    i_max, j_max = largest_square_endpoint

    # If the square length is not even, reduce by 1
    if max_square_length % 2 == 1:
        max_square_length -= 1

    square_data = input_data[:, i_max - max_square_length:i_max, j_max - max_square_length:j_max]

    return square_data


def img_to_vec(image_data, mask):
    """Converts from a 2-D array (an "image") to a row vector in raster order, using a boolean array called a "mask"
    that indicates the pixels in the image which should be included in the vector. The function can convert either a
    single image to a single vector or a 3-D array (containing a sequence of images) to a sequence of rows vectors
    (i.e., to the rows of a 2-D array).

    Args:
        image_data (ndarray): numpy 2-D or 3-D array containing the image pixel values.

            - If image_data is 2-D, a single image is converted to a single vector. In this case, the input must have
                shape (image height, image width).
            - If image_data is 3-D, a sequence of images is converted to a sequence of row vectors. In this case, the
                input must have shape (number of images, image height, image width).

        mask (ndarray): numpy 2-D boolean array of shape (image height, image width) indicating which 2-D data indices
            to include in the output vector(s).

    Returns:
        **output_vec** (*ndarray*) -- numpy 1-D or 2-D array with the (flattened) image pixel values. If "image_data" is
        2-D, only a single 1-D array is returned. If "image_data" is 3-D, a 2-D array of shape (number of images, image
        dimensionality) is returned.
    """
    assert (2 <= len(image_data.shape) <= 3)
    # Determines if we are converting a single image or a sequence of images:
    if len(image_data.shape) == 2:  # converting a single image
        assert (image_data.shape == mask.shape)
        output_vec = image_data[mask]
    else:
        assert (image_data.shape[1:] == mask.shape)
        # Converting a sequence of images
        output_vec = image_data[:, mask]

    return output_vec



def structure_function_2d(data_values, mask=None, compute_square_root=False):
    """Estimate a generalized quasi-homogeneous Kolmogorov spatial structure function of an input time-series of 2-D
    data. The spatial structure function values are the (estimated) mean-squared differences of the input array "data"
    values at pairs of 2-D spatial locations. While the standard structure function computes these values as a function
    just the of relative separation (or the pixel distance between two spatial locations), the anisotrpic structure
    function depends on two variables: the relative separation and the angle of the difference between the two spatial
    locations.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data from which we would like to calculate turbulence structure function values.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        compute_square_root (bool, optional): [Default=False] choice of whether to compute (and return) the square root
            of the structure function values (instead of the structure function values themselves)

    Returns:
        - **structure_function_inputs** (*ndarray*) -- numpy 2-D array of shape (number of inputs, 2) containing each
          (relative separation, angle) input to the structure function.
        - **structure_function** (*ndarray*) -- numpy 1-D array of shape (number of inputs,) containing the estimated
          structure function values (in the same order as the first output).
    """
    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data_values)), axis=0) == 1)

    # Normalize the data's statistics by removing the moving and dividing by the standard deviation
    # (along each pixel):
    data_flat = img_to_vec(image_data=data_values, mask=mask).T
    mean = np.average(data_flat, axis=1)
    data_mean_removed = data_flat - mean[:, np.newaxis]
    std_dev = np.linalg.norm(data_mean_removed, axis=1) / np.sqrt(data_values.shape[0])
    data_normalized = data_mean_removed / std_dev[:, np.newaxis]

    # Calculate the list of relative separations between pixels (units: number of pixels)
    spatial_indices = np.argwhere(mask)
    differences = spatial_indices[:, np.newaxis, :] - spatial_indices[np.newaxis, :, :]
    relative_separations = np.linalg.norm(differences, axis=2)
    # Extract unique relative separations
    relative_separations = relative_separations[np.triu_indices(n=relative_separations.shape[0], k=1)]

    # Compute the quasi-homogeneous spatial structure function using the normalized data
    spatial_covariance = (1.0 / data_values.shape[0]) * (data_normalized @ data_normalized.T)
    # Extract the spatial covariance values that we need for structure function calculations
    covariance_values = spatial_covariance[np.triu_indices(n=spatial_covariance.shape[0], k=1)]
    structure_function_values = 2 * (1 - covariance_values)

    # Take the square root of all structure function values (if prompted to by the user):
    if compute_square_root:
        structure_function_values = np.sqrt(structure_function_values)

    # Sort the relative separation values in ascending order
    sort_indices = np.argsort(relative_separations)
    relative_separations = relative_separations[sort_indices]

    # Sort the structure function array accordining to the same indices
    structure_function_values = structure_function_values[sort_indices]

    # Compute and sort the angle of each difference
    angles = np.arctan2(differences[:, :, 0], differences[:, :, 1])
    angles = angles[np.triu_indices(n=angles.shape[0], k=1)]
    angles = np.mod(angles[sort_indices], np.pi)

    # Average the structure function values of each (relative separation, angle) pair
    unique_relative_separations = np.unique(relative_separations)
    structure_function_inputs = []
    structure_function = []
    for index, relative_separation in enumerate(unique_relative_separations):
        relative_separation_indices = np.squeeze(np.argwhere(relative_separations == relative_separation))
        associated_angles = angles[relative_separation_indices]
        unique_associated_angles = np.sort(np.unique(associated_angles))
        for angle in unique_associated_angles:
            angle_indices = np.squeeze(np.argwhere(angles == angle))
            intersect_indices = np.intersect1d(relative_separation_indices, angle_indices)
            structure_function_inputs.append([relative_separation, angle])
            structure_function.append(np.average(structure_function_values[intersect_indices]))

    structure_function_inputs = np.array(structure_function_inputs)
    structure_function = np.array(structure_function)

    return structure_function_inputs, structure_function
