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
ASPECT_RATIO = 0.75             # aspect ratio of image (num_rows / num_columns)


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


def form_img_from_projections(projections):
    """
    Form an image from a number of projections at different angles (sinogram), i.e. perform reverse radon transform
        only a single channel should be passed at a time
    :param projections: np.array of projections
    :return: image
    """
    num_angles, num_sensors = projections.shape
    image = np.zeros((int(ASPECT_RATIO * num_sensors), num_sensors))
    delta_theta = 180 / num_angles
    back_projections = np.broadcast_to(projections, (int(ASPECT_RATIO * num_sensors), *projections.shape))
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
    img = skimage.io.imread(IMG_FILE)   # Load image
    pool = ThreadPool(img.shape[-1])    # Create a separate thread for each image channel
    funcs_to_do = [                     # List of functions to preform on each channel (in order)
        apply_fft_to_projections,
        apply_ramp_filter,
        apply_inverse_fft_to_projections,
        form_img_from_projections,
    ]
    img = np.rollaxis(img, 2, 0)  # Make images channel first to allow iterating over channels
    final_image = pool.starmap(apply_functions, zip(img, repeat(funcs_to_do)))
    final_image = np.rollaxis(np.array(final_image), 0, 3)  # Make image channels last for displaying
    plt.imshow(final_image)
    plt.show()
