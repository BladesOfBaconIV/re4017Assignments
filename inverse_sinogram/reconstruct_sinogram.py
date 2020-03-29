# Assignment 1: Reconstruct image from sinogram
# Python 3.7

# TODO implement hamming / hann windowing

import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transforms
import skimage
from multiprocessing import Pool as ThreadPool
from itertools import repeat

IMG_FILE = "./sinogram.png"     # image file to open
ASPECT_RATIO = 4/3              # aspect ratio of image (width:height)
USE_MULTI_THREADING = True      # Whether to use multi-threading to process channels simultaneously


def apply_fft_to_projections(projections):
    """
    Apply a fast fourier transform to each projection
    :param projections: np.array of projections
    :return: np.array of fft(projection)
    """
    return fft.rfft(projections, axis=1)


def apply_ramp_filter(projection_ffts):
    """
    Apply a ramp filter to each row in an array
        rows should be in frequency domain to allow for multiplication instead of convolution
    :param projection_ffts: np.array of fft of projections
    :return: np.array of ramp_filter * projections
    """
    ramp = np.floor(np.arange(0.5, projection_ffts.shape[1] // 2 + 0.1, 0.5))
    return projection_ffts * ramp


def apply_inverse_fft_to_projections(projection_ffts):
    """
    Apply the inverse fft to rows in an array
    :param projection_ffts: np.array of fft of projections
    :return: np.array of projections
    """
    return fft.irfft(projection_ffts, axis=1)


def form_img_from_projections(projections, rescale=True):
    """
    Form an image from a number of projections at different angles (sinogram), i.e. perform reverse radon transform
        only a single channel should be passed at a time
    :param projections: np.array of projections
    :return: image
    """
    num_angles, num_sensors = projections.shape
    image = np.zeros((num_sensors, num_sensors))
    delta_theta = 180 / num_angles
    back_projections = np.broadcast_to(projections, (num_sensors, *projections.shape))
    for i in range(num_angles):
        image += transforms.rotate(back_projections[:, i, :], delta_theta*i)
    return image


def apply_functions(projections, functions):
    """
    Create a pipeline of functions to apply to an array of projections
    :param projections: np.array of projections (sinogram)
    :param functions: ordered list of functions to apply,
                        result of previous function will be fed into the next function
    :return: final result of pipeline
    """
    for func in functions:
        projections = func(projections)
    return projections


if __name__ == "__main__":
    sinogram = skimage.io.imread(IMG_FILE)   # Load image
    pool = ThreadPool(sinogram.shape[-1])    # Create a separate thread for each image channel
    funcs_to_do = [                     # List of functions to preform on each channel (in order)
        apply_fft_to_projections,
        apply_ramp_filter,
        apply_inverse_fft_to_projections,
        form_img_from_projections,
    ]
    # Process projections
    channels = [sinogram[:, :, i] for i in range(sinogram.shape[-1])]
    if USE_MULTI_THREADING:
        channel_images = pool.starmap(apply_functions, zip(channels, repeat(funcs_to_do)))
    else:
        channel_images = [apply_functions(c, funcs_to_do) for c in channels]

    # Calculate points to crop image ( Desired image must be within a circle with same width as sinogram )
    r = sinogram.shape[1]                   # Radius of image circle
    alpha = np.arctan(1/ASPECT_RATIO)       # Find angle to corner that gives rectangle of aspect ratio
    final_width, final_height = np.floor([r * np.cos(alpha), r * np.sin(alpha)]).astype('int32')  # Wanted width and height
    r, c = channel_images[0].shape                   # current rows and columns of channel images
    start_r = int((r // 2) - (final_height // 2))       # where to start cropping rows
    start_c = int((c // 2) - (final_width // 2))        # where to start cropping columns

    for i in range(len(channel_images)):
        channel_images[i] = channel_images[i][start_r:start_r+final_height, start_c:start_c+final_width]
        # Rescale pixel values
        clo, chi = channel_images[i].min(), channel_images[i].max()
        channel_images[i] = 255 * (channel_images[i] - clo) / (chi - clo)
        channel_images[i] = np.floor(channel_images[i]).astype('uint8')

    # Combine channels to make final image
    final_image = np.dstack(channel_images)
    plt.imshow(final_image)
    # Crop image
    plt.show()
