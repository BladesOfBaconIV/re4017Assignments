# Assignment 1: Reconstruct image from sinogram
# Python 3.7

import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transforms
import skimage

IMG_FILE = "./sinogram.png"


def apply_fft_to_projections(projections):
    return fft.rfft(projections, axis=1)


def apply_ramp_filter(proj_ffts):
    ramp = np.floor(np.arange(0.5, proj_ffts.shape[1]//2 + 0.1, 0.5))
    return proj_ffts * ramp


def apply_inverse_fft_to_projections(projections):
    return fft.irfft(projections, axis=1)


def form_img_from_projections(projections):
    num_angles, num_sensors = projections.shape
    image = np.zeros((num_sensors, num_sensors))
    delta_theta = 180 / num_angles
    for i in range(num_angles):
        print(i)
        back_projection = np.tile(projections[i], (num_sensors, 1))
        image += transforms.rotate(back_projection, delta_theta*i)
    return image


if __name__=="__main__":
    img = skimage.io.imread(IMG_FILE)
    print(img.shape)
    num_projections, num_sensors, num_channels = img.shape
    final_image = np.zeros((num_sensors, num_sensors, num_channels))
    for channel in range(num_channels):
        print(f'##########\nChannel {channel}\n##########')
        p = img[:, :, channel]
        p_fft = apply_fft_to_projections(p)
        p_fft_filtered = apply_ramp_filter(p_fft)
        p_processed = apply_inverse_fft_to_projections(p_fft_filtered)
        channel_img = form_img_from_projections(p_processed)
        final_image[:, :, channel] += channel_img
    plt.imshow(final_image)