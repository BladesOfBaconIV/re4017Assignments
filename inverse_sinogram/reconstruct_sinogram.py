# Assignment 1: Reconstruct image from sinogram
# Authors:
#   Darragh Glavin:     16189183
#   Lorcan Williamson:  16160703
#   Yilin Mou:          18111602
# Python 3.7

import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transforms
from skimage.io import imread
from multiprocessing import Pool as ThreadPool
from itertools import repeat

IMG_FILE = "./sinogram.png"     # image file to open
ASPECT_RATIO = 4/3              # aspect ratio of image (width:height)
USE_MULTI_THREADING = True      # Whether to use multi-threading to process channels simultaneously
SAVE_INTERMEDIATE = False       # Whether to save intermediate images during processing (helpful for writing report)


def apply_fft(sinogram):
    """
    Apply a fast fourier transform to each projection
    :param sinogram: np.array of projections
    :return: np.array of fft(projection)
    """
    return fft.rfft(sinogram, axis=1)


def apply_ramp_filter(sinogram_fft):
    """
    Apply a frequency domain ramp filter to sinogram_fft
    :param sinogram_fft: np.array of projection ffts
    :return: apply_filter(sinogram_fft) function
    """
    # Applies a filter to the fft of a sinogram
    ramp = np.floor(np.arange(0.5, sinogram_fft.shape[1] // 2 + 0.1, 0.5))
    return sinogram_fft * ramp


def apply_hamming_windowed_ramp_filter(sinogram_fft):
    """
    Apply a frequency domain, hamming windowed, ramp filter to sinogram_fft
    :param sinogram_fft: np.array of projection ffts
    :return: apply_filter(sinogram_fft) function
    """
    # Applies a filter to the fft of a sinogram
    ramp = np.floor(np.arange(0.5, sinogram_fft.shape[1] // 2 + 0.1, 0.5))
    hamming = 0.54 + 0.46 * np.cos((np.pi * ramp) / ramp[-1])
    return sinogram_fft * ramp * hamming


def apply_hann_windowed_ramp_filter(sinogram_fft):
    """
    Apply a frequency domain, hann windowed, ramp filter to sinogram_fft
    :param sinogram_fft: np.array of projection ffts
    :return: apply_filter(sinogram_fft) function
    """
    # Applies a filter to the fft of a sinogram
    ramp = np.floor(np.arange(0.5, sinogram_fft.shape[1] // 2 + 0.1, 0.5))
    hann = 0.5 + 0.5 * np.cos((np.pi * ramp) / ramp[-1])
    return sinogram_fft * ramp * hann


def apply_inverse_fft(sinogram_fft):
    """
    Apply the inverse fft to rows in an array
    :param sinogram_fft: np.array of fft of projections
    :return: np.array of projections
    """
    return fft.irfft(sinogram_fft, axis=1)


def form_img_from_sinogram(sinogram):
    """
    Form an image from a number of projections at different angles (sinogram), i.e. perform reverse radon transform
        only a single channel should be passed at a time
    :param sinogram: np.array of projections
    :return: image
    """
    num_angles, num_sensors = sinogram.shape
    image = np.zeros((num_sensors, num_sensors))
    delta_theta = 180 / num_angles
    for i in range(num_angles):
        back_projection = np.tile(sinogram[i, :], (num_sensors, 1))
        image += transforms.rotate(back_projection, delta_theta*i)
    return image


def crop(channel):
    """
    Crop an image channel to the global aspect ratio
        Assumes image was made from a sinogram
    :param channel: single channel from the image
    :return: cropped image channel
    """
    r, c = channel.shape  # Current rows and columns of channel images
    final_height = int(r / np.sqrt(1 + ASPECT_RATIO ** 2))  # Width of original image
    final_width = int(final_height * ASPECT_RATIO)  # Height of original image
    start_r = (r // 2) - (final_height // 2)  # where to start cropping rows
    start_c = (c // 2) - (final_width // 2)   # where to start cropping columns
    return channel[start_r:start_r + final_height, start_c:start_c + final_width]


def rescale(channel):
    """
    Rescale image channel pixel values back to range 0...255
    :param channel: image channel to rescale
    :return: rescaled image channel
    """
    clo, chi = channel.min(), channel.max()
    return np.floor(255 * (channel - clo) / (chi - clo)).astype('uint8')


def apply_functions(sinogram, functions, save_intermediate=False):
    """
    Create a pipeline of functions to apply to an array of projections
    :param sinogram: np.array of projections
    :param functions: ordered list of functions to apply,
                        result of previous function will be fed into the next function
    :param save_intermediate: save the intermediate values
    :return: final result of pipeline, or list of intermediate values (including final value)
    """
    intermediate = [sinogram] if save_intermediate else None
    for func in functions:
        sinogram = func(sinogram)
        if save_intermediate:
            intermediate.append(sinogram)
    return intermediate if save_intermediate else sinogram


if __name__ == "__main__":
    rgb_sinogram = imread(IMG_FILE)      # Load image
    pool = ThreadPool(rgb_sinogram.shape[-1])       # Create a separate thread for each image channel
    funcs_to_do = [                                 # List of functions to preform on each channel (in order)
        apply_fft,
        apply_hann_windowed_ramp_filter,
        apply_inverse_fft,
        form_img_from_sinogram,
        crop,
        rescale,
    ]

    # Process projections
    channels = [rgb_sinogram[:, :, i] for i in range(rgb_sinogram.shape[-1])]
    if USE_MULTI_THREADING:
        channel_images = pool.starmap(apply_functions, zip(channels, repeat(funcs_to_do), repeat(SAVE_INTERMEDIATE)))
    else:
        channel_images = [apply_functions(c, funcs_to_do, SAVE_INTERMEDIATE) for c in channels]

    # Display images
    if SAVE_INTERMEDIATE:
        for result in zip(*channel_images):
            plt.figure()
            plt.imshow(rescale(np.dstack(result)))
            plt.show()
    else:
        plt.imshow(np.dstack(channel_images))
        plt.show()
