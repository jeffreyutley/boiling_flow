import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation
import boiling_flow

# Approved for public release; distribution is unlimited. Public Affairs release approval #2025-5580.

def slopes_psd(data_values, locations=None, axis=2, time_block_size=1024, sampling_frequency=None, remove_mean=True,
               use_overlapping_blocks=True):
    """Approximates the slopes (gradient) of the data values (using a second order finite difference method) with
    respect to a user-specified axis and uses temporal_psd() to estimate the temporal power spectral density (PSD) of
    these slopes.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values.
        locations (ndarray, optional): [Default=None] numpy 1-D array containing the coordinates of the axis to which
            the gradient is to be calculated. Sent as an argument to np.gradient.

            - If set to None, this argument is not included in the call to np.gradient. The indices are instead used.

        axis (int, optional): [Default=2] the axis of data values to take the gradient with respect to.

            - If axis = 2, the gradient is calculated with respect to the x-axis.

            - If axis = 1, the gradient is calculated with respect to the y-axis.

            - If axis = 0, the gradient is calculated with respect to time.

        time_block_size (int, optional): [Default=1000] the size of each time block to use for the PSD approximation.
            The full time-series is broken up into distinct "time-blocks" of the indicated size. For each time-block,
            the PSD is calculated independently. The final PSD calculation is then the average over each time-block.
            This value must be a positive even integer and can be at most the number of time-steps in data_values. The
            parameter is sent as an argument to the function temporal_psd().

        sampling_frequency (float, optional): [Default=None] the sampling frequency of the discrete-time signal
            "data_values." This input should be included if the desired PSD units are energy per unit time instead of
            energy per unit sample. In this case, the frequency bins are in units of cycles per unit time instead of
            cycles per unit sample. This value is sent as an argument to the function temporal_psd().

            - If set to None, the PSD units are energy/ample and the frequency units are cycles/sample.

        remove_mean (bool, optional): [Default=True] choice of removing the temporal mean of each vector component
            before computing the PSD. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

        use_overlapping_blocks (bool, optional): [Default=True] whether to use overlapping time-blocks when calculating
            the PSD. If set to True, then the time-blocks will have a 50% overlap. This method allows one to maintain
            the same block size while also reducing noise in the PSD calculation. If set to False, then the time-blocks
            will have no overlap.

    Returns:
        - **frequencies** (*ndarray*) -- numpy 1-D array containing the frequency bins of the PSD calculation.
        - **slopes_psd_estimate** (*ndarray*) -- numpy 1-D array containing the PSD estimates for each frequency bin.
    """
    assert (axis in [0, 1, 2])

    if locations is None:
        data_slopes = np.gradient(data_values, axis=axis)
    else:
        data_slopes = np.gradient(data_values, locations, axis=axis)

    # Sets the mask to be the intersection of valid data values for all images in the time-series:
    mask = (np.average(1 - np.uint8(np.isnan(data_slopes)), axis=0) == 1)
    frequencies, slopes_psd_estimate = temporal_psd(data_values=data_slopes, time_block_size=time_block_size, mask=mask,
                                                    sampling_frequency=sampling_frequency, remove_mean=remove_mean,
                                                    use_overlapping_blocks=use_overlapping_blocks)
    return frequencies, slopes_psd_estimate


def temporal_psd(data_values, time_block_size=1024, mask=None, sampling_frequency=None, remove_mean=True,
                 use_overlapping_blocks=True):
    """Approximates the temporal Power Spectral Density (PSD) of the input data values by averaging the 1-D PSD
    estimates for each pixel in the image (i.e., for each time-sequence of data values at a single pixel). Each 1-D PSD
    is estimated using Welch's method, in which the time-series is broken up into independent "blocks" of length
    "time_block_size." A Hamming window is applied to each block and the PSD is estimated using an FFT and a scaling
    factor which lowers the variance of the estimate.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values
        time_block_size (int, optional): [Default=1024] the size of each time block to use for the PSD estimation.
            The full time-series is broken up into distinct "time-blocks" of the indicated size. For each time-block,
            the PSD is calculated independently. The final PSD calculation is then the average over each time-block.
            This value must be a positive even integer and can be at most the number of time-steps in data_values.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        sampling_frequency (float, optional): [Default=None] the sampling frequency of the discrete-time signal
            "data_values." This input should be included if the desired PSD units are energy per unit time instead of
            energy per unit sample. In this case, the frequency bins are in units of cycles per unit time instead of
            cycles per unit sample.

            - If set to None, the PSD units are energy/ample and the frequency units are cycles/sample.

        remove_mean (bool, optional): [Default=True] choice of removing the temporal mean of each vector component
            before computing the PSD. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

        use_overlapping_blocks (bool, optional): [Default=True] whether to use overlapping time-blocks when calculating
            the PSD. If set to True, then the time-blocks will have a 50% overlap. This method allows one to maintain
            the same block size while also reducing noise in the PSD calculation. If set to False, then the time-blocks
            will have no overlap.

    Returns:
        - **frequencies** (*ndarray*) -- numpy 1-D array containing the frequency bins of the PSD calculation.
        - **psd_estimate** (*ndarray*) -- numpy 1-D array containing the PSD estimates for each frequency bin.
    """
    assert (len(data_values.shape) == 3)
    assert (0 < time_block_size <= data_values.shape[0])

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data_values)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data_values.shape[1], data_values.shape[2]))
        assert (not np.isnan(boiling_flow.utils.img_to_vec(image_data=data_values, mask=mask)).any())

    # Converts the input OPD images to vectors:
    data_vector = boiling_flow.utils.img_to_vec(image_data=data_values, mask=mask)

    # Calculates the number of time-blocks we average over:
    if use_overlapping_blocks:
        num_blocks = 2 * (data_vector.shape[0] // time_block_size) - 1
    else:
        num_blocks = data_vector.shape[0] // time_block_size

    # Initializes a Hamming window and computes relevant quantities:
    hamming_window = np.hamming(time_block_size)
    hamming_window_sum = np.sum(hamming_window)
    welch_scaling_factor = np.sum(hamming_window ** 2)

    # Frequencies along the x-axis:
    frequencies = np.fft.rfftfreq(n=time_block_size)

    # Removes the temporal mean from each vector component:
    if remove_mean:
        temporal_mean = np.mean(data_vector, axis=0)
        data_vector = data_vector - temporal_mean[np.newaxis, :]

    # Iterates over each time-block and averages the temporal PSD:
    psd_estimates = np.zeros((len(frequencies), data_vector.shape[1]))
    for block_idx in range(num_blocks):
        # Extracts the time-series for the current block:
        if use_overlapping_blocks:
            block_data = data_vector[int((block_idx / 2) * time_block_size):
                                     int(((block_idx / 2) + 1) * time_block_size)]
        else:
            block_data = data_vector[block_idx * time_block_size: (block_idx + 1) * time_block_size]

        # If mean-removal is not selected, takes the FFT of the windowed block data (with its mean):
        windowed_data = block_data * hamming_window[:, np.newaxis]

        # Uses an FFT to approximate the DFT of the current section:
        block_dft = np.fft.rfft(windowed_data, axis=0)

        # Uses the FFT to approximate the energy spectrum of the current time-block:
        block_energy_spectrum = np.abs(block_dft) ** 2

        # Applies the Welch's method scaling factor to compute the PSD estimate for this block
        psd_estimates += block_energy_spectrum / welch_scaling_factor

    # Averages the PSD estimates across each row (i.e., across the estimates for each pixel) and divides by the number
    # of sections used:
    psd_estimate = np.average(psd_estimates, axis=1) / num_blocks

    # If a sampling frequency is provided, divides the PSD estimate by the sampling frequency to ensure correct unit
    # conversion:
    if sampling_frequency:
        assert (sampling_frequency > 0)
        frequencies = sampling_frequency * frequencies
        psd_estimate = psd_estimate / sampling_frequency

    return frequencies, psd_estimate


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
    data_flat = boiling_flow.utils.img_to_vec(image_data=data_values, mask=mask).T
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


def compute_rms(data_values, mask=None):
    """Take the root mean-square (RMS) of a time-series of 2-D arrays.

    Args:
        data_values (ndarray): numpy 2-D or 3-D array of shape (M, N) or (num_time_steps, M, N) containing the data
        mask (ndarray, optional): [Default=None] numpy boolean 2-D of shape (M, N) specifying which pixels to use

            - If set to None, all pixels are used.

    Returns:
        **rms** (*float*) -- the RMS of the data values
    """
    assert (data_values.ndim in [2, 3])
    if mask is not None:
        assert (mask.shape == data_values.shape[1:])
        assert (mask.dtype == bool)
    else:
        mask = np.ones(data_values.shape[1:], dtype=bool)

    data_flat = boiling_flow.utils.img_to_vec(data_values, mask)
    rms = np.sqrt(np.average(data_flat ** 2))
    return rms


def compute_nrmse(ground_truth_data, estimated_data):
    """Compute the normalized root mean-square error (NRMSE) of estimated data with respect to ground-truth data. We use
    range normalization of the RMSE. The range is difference between the 95th and 5th percentiles of the ground-truth
    data.

    Args:
        ground_truth_data (ndarray): numpy array containing the ground-truth values
        estimated_data (ndarray): numpy array containing the estimated values

    Returns:
        **nrmse** (*float*) -- the NRMSE between the estimated and ground-truth data values
    """
    data_error = ground_truth_data - estimated_data
    rmse = np.sqrt(np.average(data_error ** 2))
    percentile_95 = np.percentile(ground_truth_data, 95)
    percentile_5 = np.percentile(ground_truth_data, 5)
    value_range = percentile_95 - percentile_5
    nrmse = rmse / value_range
    return nrmse


def plot_tps(frequencies, tps_values, xlabel='', ylabel='', title='', tps_values2=None, label1='', label2='',
             savefile=None,  show=True):
    """Plot TPS values and corresponding frequencies.

    Args:
        frequencies (ndarray): numpy 1-D array containing the frequencies
        tps_values (ndarray): numpy 1-D array containing the TPS values
        xlabel (str, optional): [Default=''] label for the x-axis
        ylabel (str, optional): [Default=''] label for the y-axis
        title (str, optional): [Default=''] title for the plot
        tps_values2 (ndarray, optional): [Default=None] numpy 1-D array containing a second TPS

            - If set to None, only tps_values is plotted.

        label1 (str, optional): [Default=''] label for the first TPS
        label2 (str, optional): [Default=''] label for the second TPS
        savefile (str, optional): [Default=None] directory at which to save the figure

            - If set to None, no figure is saved.

        show (bool, optional): [Default=True] whether to show the figure

    Returns:
        **fig** (*matplotlib.figure.Figure*) -- the figure created by the function
    """
    fig = plt.figure()

    if tps_values2 is None:
        plt.plot(frequencies, tps_values, '.')
        plt.plot(frequencies, tps_values, 'r', linewidth=0.5)
    else:
        plt.plot(frequencies, tps_values, '-o', label=label1, markersize=5, markerfacecolor='green', linewidth=0.5)
        plt.plot(frequencies, tps_values2, '-o', label=label2, markersize=5, markerfacecolor='red', linewidth=0.5)
        plt.legend(fontsize=13)

    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig


def interpolate_image(x, y, image, x_pixel_length, y_pixel_length, num_x_pixels, num_y_pixels,
                      add_zero=True):
    """Interpolate a 2-D array to create an image.

    Args:
        x (ndarray): numpy 1-D array of x-axis values
        y (ndarray): numpy 1-D array of y-axis values
        image (ndarray): numpy 2-D array containing the image
        x_pixel_length (float): length of x-axis pixels
        y_pixel_length (float): length of y-axis pixels
        num_x_pixels (int): number of x-axis pixels
        num_y_pixels (int): number of y-axis pixels
        add_zero (bool, optional): [Default=True] whether to add a value of zero at (0, 0) to the interpolation

    Returns:
        **image_interpolated** (*ndarray*) -- numpy 2-D array containing the interpolated image
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_values = np.linspace(x_min + 0.5*x_pixel_length, x_max - 0.5*x_pixel_length, num_x_pixels)
    y_values = np.linspace(y_min + 0.5*y_pixel_length, y_max - 0.5*y_pixel_length, num_y_pixels)

    if add_zero:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        image = np.insert(image, 0, 0)

    X, Y = np.meshgrid(x_values, y_values)

    image_interpolated = scipy.interpolate.griddata((x, y), image, (X, Y), method='linear')

    lowest_distance = np.min(np.sqrt(X ** 2 + Y ** 2))

    close_to_zero_indices = np.argwhere(np.abs(np.sqrt(X ** 2 + Y ** 2) - lowest_distance) < 1e-10)

    for i in range(close_to_zero_indices.shape[0]):
        zero_index = (close_to_zero_indices[i][0], close_to_zero_indices[i][1])
        image_interpolated[zero_index] = np.nan

    return x_values, y_values, image_interpolated


def extend_angles(structure_function_inputs, structure_function):
    """Extend the structure function data to have angles in the range [0, 2*pi).

    Args:
        structure_function_inputs (ndarray): numpy 2-D array containing the structure function inputs (in polar
            coordinates)
        structure_function (ndarray): numpy 2-D array containing the structure function values

    Returns:
        - **new_structure_function_inputs** (*ndarray*) -- numpy 2-D array containing the extended structure function
          inputs
        - **new_structure_function** (*ndarray*) -- numpy 2-D array containing the extended structure function values
    """
    additive_pi = np.tile(np.array([0, np.pi]), (structure_function_inputs.shape[0], 1))
    additional_structure_function_inputs = structure_function_inputs + additive_pi

    new_structure_function_inputs = np.zeros((2 * structure_function_inputs.shape[0], 2))
    new_structure_function = np.zeros(2 * structure_function.shape[0])

    new_structure_function_inputs[:structure_function_inputs.shape[0]] = structure_function_inputs
    new_structure_function_inputs[structure_function_inputs.shape[0]:] = additional_structure_function_inputs

    new_structure_function[:structure_function_inputs.shape[0]] = structure_function
    new_structure_function[structure_function_inputs.shape[0]:] = structure_function

    return new_structure_function_inputs, new_structure_function


def polar_to_rectangular(structure_function_inputs):
    """Convert the structure function inputs from polar to rectangular coordinates.

    Args:
        structure_function_inputs (ndarray): numpy 2-D array containing the structure function inputs (in polar
            coordinates)

    Returns:
        - **x** (*ndarray*) -- numpy 1-D array containing the x-coordinates
        - **y** (*ndarray*) -- numpy 1-D array containing the y-coordinates
    """
    num_points = structure_function_inputs.shape[0]
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    for point_index in range(num_points):
        r = structure_function_inputs[point_index, 0]
        theta = structure_function_inputs[point_index, 1]
        x[point_index], y[point_index] = (r * np.cos(theta), r * np.sin(theta))
    return x, y


def structure_function_image(structure_function_inputs, structure_function_values, title='', x_label='', y_label='',
                             cbar_label='', structure_function_values_2=None, title_2='', suptitle='', savefile=None,
                             show=True):
    """Create an image of the structure function.

    Args:
        structure_function_inputs (ndarray): numpy 2-D array containing the structure function inputs  (in polar
            coordinates)
        structure_function (ndarray): numpy 2-D array containing the structure function values
        title (str, optional): [Default=''] title for the image
        xlabel (str, optional): [Default=''] label for the x-axis
        ylabel (str, optional): [Default=''] label for the y-axis
        cbar_label (str, optional): [Default=''] label for the color-bar
        structure_function_values_2 (ndarray, optional): [Default=None] numpy 2-D array containing another structure
            function

            - If set to None, only structure_function_values is plotted.

        title_2 (str, optional): [Default=''] title for the second structure function image
        savefile (str, optional): [Default=None] directory at which to save the figure

            - If set to None, no figure is saved.

        show (bool, optional): [Default=True] whether to show the figure

    Returns:
        - **fig** (*matplotlib.figure.Figure*) -- the figure created by the function
        - **axes** (*matplotlib.axes.Axes*) -- the axes created by the function
    """
    extended_inputs, structure_function_values = extend_angles(structure_function_inputs, structure_function_values)
    x, y = polar_to_rectangular(extended_inputs)
    num_x_pixels = int(x.max() - x.min() + 1)
    num_y_pixels = int(y.max() - y.min() + 1)

    x_values, y_values, structure_function_image = interpolate_image(x, y, structure_function_values, 1,
                                                                     1, num_x_pixels, num_y_pixels,
                                                                     add_zero=True)
    mask = ~np.isnan(structure_function_image)

    if structure_function_values_2 is not None:
        structure_function_values_2 = extend_angles(structure_function_inputs, structure_function_values_2)[1]

        structure_function_image_2 = interpolate_image(x, y, structure_function_values_2, 1, 1,
                                                       num_x_pixels, num_y_pixels, add_zero=True)[-1]
        v_min = np.array([structure_function_image[mask].min(), structure_function_image_2[mask].min()]).min()
        v_max = np.array([structure_function_image[mask].max(), structure_function_image_2[mask].max()]).max()

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        im = axes[0].imshow(structure_function_image, interpolation='none', aspect='equal', origin='lower',
                            vmin=v_min, vmax=v_max,
                            extent=[x_values[0] - 0.5, x_values[-1] + 0.5,
                                    y_values[0] - 0.5, y_values[-1] + 0.5])
        axes[0].set_title(title)
        im = axes[1].imshow(structure_function_image_2, interpolation='none', aspect='equal', origin='lower',
                            vmin=v_min, vmax=v_max,
                            extent=[x_values[0] - 0.5, x_values[-1] + 0.5,
                                    y_values[0] - 0.5, y_values[-1] + 0.5])
        axes[1].set_title(title_2)
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(y_label)
        axes[1].set_xlabel(x_label)
    else:
        v_min = structure_function_image[mask].min()
        v_max = structure_function_image[mask].max()

        fig = plt.figure()
        axes = plt.axes()
        im = axes.imshow(structure_function_image, interpolation='none', aspect='equal', origin='lower',
                            vmin=v_min, vmax=v_max,
                            extent=[x_values[0] - 0.5, x_values[-1] + 0.5,
                                    y_values[0] - 0.5, y_values[-1] + 0.5])
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)

    color_bar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, shrink=0.6)
    color_bar.set_label(cbar_label)

    plt.suptitle(suptitle)

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig, axes


def create_video(data, title='', mask=None, data2=None, title2=''):
    """Creates a video from a sequence of images using matplotlib.pyplot and matplotlib.animation. Shows each image on
    the same scale and includes a colorbar. Each frame of the video can either have a single image or two images
    side-by-side.

    Args:
        data (ndarray): numpy 3-D array of shape (number of frames, image height, image width) containing the data
            values to create a video of.
        title (str, optional): [Default=''] title to place above the video.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        data2 (ndarray, optional): [Default=None] numpy 3-D array with the same shape as "data" and which contains
            additional data to create a video from. If included, each frame of the video will include two images
            side-by-side.

            - If set to None, only one image is included in each frame.

        title2 (str, optional): [Default=''] title above the images from "data2." Only used by the function if "data2"
            is included as an input argument.

    Returns:
        **video** (*matplotlib.animation.ArtistAnimation*) - the ArtistAnimation figure containing the video.
    """
    assert (len(data.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the sequence:
        mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data.shape[1], data.shape[2]))
        assert (not np.isnan(boiling_flow.utils.img_to_vec(image_data=data, mask=mask)).any())

    num_frames = data.shape[0]
    video_frames = []

    if data2 is not None:
        # Ensures that the mask is valid for data2:
        assert (not np.isnan(boiling_flow.utils.img_to_vec(image_data=data2, mask=mask)).any())

        fig, axs = plt.subplots(1, 2)

        # Finds the minimum and maximum values of the data for the purpose of colorbar creation:
        datavals = boiling_flow.utils.img_to_vec(image_data=data, mask=mask)
        data2vals = boiling_flow.utils.img_to_vec(image_data=data2, mask=mask)
        vmin = np.array([datavals.min(), data2vals.min()]).min()
        vmax = np.array([datavals.max(), data2vals.max()]).max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            im0 = axs[0].imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            im = axs[1].imshow(data2[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([im0, axs[0].set_title(title, loc='center'), im,
                                 axs[1].set_title(title2, loc='center')])

        # Add the colorbar:
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=0.55)

    else:
        fig = plt.figure()
        axs = plt.axes()
        datavals = boiling_flow.utils.img_to_vec(image_data=data, mask=mask)
        vmin = datavals.min()
        vmax = datavals.max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            im = plt.imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([im, plt.title(title, loc='center')])

        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=1)

    video = matplotlib.animation.ArtistAnimation(fig, video_frames)

    return video
