# Assignment 1: Reconstruct image from sinogram
# Python 3.7

import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transforms
import skimage
from skimage import io,data
from multiprocessing import Pool as ThreadPool
from itertools import repeat
import math

IMG_FILE = "./sinogram.png"     # image file to open
ASPECT_RATIO = 4/3              # aspect ratio of image (width:height)
USE_MULTI_THREADING = True      # Whether to use multi-threading to process channels simultaneously
SAVE_INTERMEDIATE = True       # Whether to save intermediate images during processing (helpful for writing report)


def apply_fft(sinogram):
    """
    Apply a fast fourier transform to each projection
    :param sinogram: np.array of projections
    :return: np.array of fft(projection)
    """
    return fft.rfft(sinogram, axis=1)


def apply_ramp_filter(sinogram_fft):
    """
    Apply a ramp filter to each row in an array
        rows should be in frequency domain to allow for multiplication instead of convolution
    :param sinogram_fft: np.array of fft of projections
    :return: np.array of ramp_filter * projections
    """
    ramp = np.floor(np.arange(0.5, sinogram_fft.shape[1] // 2 + 0.1, 0.5))
    hamming = 0.5 + (1-0.5)*np.cos(np.pi*ramp/ramp[-1])

    return sinogram_fft * ramp * hamming


def apply_hamming_window(channel):
    """
    Apply a Hamming windowed ramp filter to the projections
    :param channel: np.array of projections
    :return: np.array of windowed projections
    """
    return channel #* np.hamming(channel.shape[1])


def apply_hann_window(channel):
    """
    Apply a Hann windowed ramp filter to the projections
    :param channel: np.array of projections
    :return: np.array of windowed projections
    """
    return channel #* np.hanning(channel.shape[1])


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
    back_projections = np.broadcast_to(sinogram, (num_sensors, *sinogram.shape))
    back_projections.flags['WRITEABLE'] = True
    for i in range(num_angles):
        image += transforms.rotate(back_projections[:, i, :], delta_theta*i)
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

def rgb2hsv(r, g, b):
    r,g,b = r/255.0,g/255.0,b/255.0
    mx = max(r,g,b)
    mn = min(r,g,b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60*((g-b)/df)+360)%360
    elif mx == g:
        h = (60*((b-r)/df)+120)%360
    elif mx == b:
        h = (60*((r-g)/df)+240)%360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h/60.0
    h60f = math.floor(h60)
    hi = int(h60f)%6
    f = h60-h60f
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    r,g,b = 0,0,0
    if hi == 0: r,g,b = v,t,p
    elif hi == 1: r,g,b = q,v,p
    elif hi == 2: r,g,b = p,v,t
    elif hi == 3: r,g,b = p,q,v
    elif hi == 4: r,g,b = t,p,v
    elif hi == 5: r,g,b = v,p,q
    r,g,b = int(r * 255),int(g * 255),int(b * 255)
    return r,g,b

if __name__ == "__main__":
    rgb_sinogram = skimage.io.imread(IMG_FILE)      # Load image
    pool = ThreadPool(rgb_sinogram.shape[-1])       # Create a separate thread for each image channel
    funcs_to_do = [                                # List of functions to preform on each channel (in order)
        #apply_hann_window,
        apply_fft,
        apply_ramp_filter,
        apply_inverse_fft,
        form_img_from_sinogram,
        rescale,
        crop,
    ]

    # Process projections
    channels = [rgb_sinogram[:, :, i] for i in range(rgb_sinogram.shape[-1])]
    if USE_MULTI_THREADING:
        channel_images = pool.starmap(apply_functions, zip(channels, repeat(funcs_to_do), repeat(SAVE_INTERMEDIATE)))
    else:
        channel_images = [apply_functions(c, funcs_to_do) for c in channels]

    # Display images ( Note intermediate images will not be rescaled)
    if SAVE_INTERMEDIATE:
        Number = 1
        for result in zip(*channel_images):
            plt.figure(Number)
            plt.imshow(np.dstack(result))
            Number = Number + 1
        plt.show()
    else:
        plt.imshow(np.dstack(channel_images))
        plt.show()
    for i in range(channel_images[0][6].shape[0]):
            for j in range(channel_images[0][6].shape[1]):
                h,s,v = rgb2hsv(channel_images[0][6][i][j],channel_images[1][6][i][j],channel_images[2][6][i][j])
                s = 1.33*s
                v = 0.9*v
                channel_images[0][6][i][j],channel_images[1][6][i][j],channel_images[2][6][i][j] = hsv2rgb(h,s,v)

    R = channel_images[0][6]
    G = channel_images[1][6]
    B = channel_images[2][6]
    plt.figure(8)
    plt.imshow(np.dstack([R,G,B]))
    plt.show()
