#!/usr/bin/env python3

import argparse
import numpy as np
import itk

import matplotlib.pyplot as plt
from scipy.fftpack import fftshift,fft,ifft,rfft,irfft
import torch

class RampFilter:
    r"""Implementation of the Ramp filter :math:`\Pi(\omega) = |\omega|`
    """
    def __init__(self):
        return
    def __call__(self, w):
        return torch.abs(w)


class HammingFilter:
    r"""Implementation of the Hamming filter given by :math:`\Pi(\omega) = \frac{1}{2}\left(1+\cos\left(\frac{\pi(|\omega|-\omega_L)}{\omega_H-\omega_L} \right)\right)` for :math:`\omega_L \leq |\omega| < \omega_H` and :math:`\Pi(\omega) = 1` for :math:`|\omega| \leq \omega_L` and :math:`\Pi(\omega) = 0` for :math:`|\omega|>\omega_H`. Arguments ``wl`` and ``wh`` should be expressed as fractions of the Nyquist frequency (i.e. ``wh=0.93`` represents 93% the Nyquist frequency).
    """

    def __init__(self, wl, wh):
        self.wl = wl / 2  # units of Nyquist Frequency
        self.wh = wh / 2

    def __call__(self, w):
        w = w.cpu().numpy()
        filter = np.piecewise(
            w,
            [np.abs(w) <= self.wl, (self.wl < np.abs(w)) * (self.wh >= np.abs(w)), np.abs(w) > self.wh],
            [lambda w: 1, lambda w: 1 / 2 * (1 + np.cos(np.pi * (np.abs(w) - self.wl) / (self.wh - self.wl))),
             lambda w: 0])
        return torch.tensor(filter)


def filter_projection(sinogram):  ##ramp filter
    a = 0.1
    num_thetas,size_x,size_y = sinogram.shape
    step = 2 * np.pi / size_x
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < size_x:
        w = np.concatenate([w, w[-1] + step])
    rn1 = np.abs(2 / a * np.sin(a * w / 2))
    rn2 = np.sin(a * w / 2) / (a * w / 2)
    r = rn1 * (rn2) ** 2

    filter_ = fftshift(r)

    # filter_ = [np.abs(x)

    fig,ax = plt.subplots()
    ax.plot(filter_)
    plt.show()

    filter_sinogram = np.zeros((num_thetas,size_x,size_y))
    for i in range(size_x):
        for angle in range(num_thetas):
            proj_fft = fft(sinogram[angle,i,:])
            # print(proj_fft.shape)
            filter_proj = proj_fft * filter_
            filter_sinogram[angle,i,:] = np.real(ifft(filter_proj))
    return filter_sinogram


"Translate the sinogram to the frequency domain using Fourier Transform"
def fft_projs(projs_1d):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return rfft(projs_1d)



"Filter the projections using a ramp filter"
def ramp_filter(fft_sino):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, fft_sino.shape[0]//2 + 0.1, 0.5))
    return fft_sino * ramp

"Return to the spatial domain using inverse Fourier Transform"
def inverse_fft_translate(operator):
    return irfft(operator)



def main():
    print(args)

    sinogram = itk.imread(args.input)
    sinogram_array = itk.array_from_image(sinogram)

    # filtered_sinogram_array = filter_projection(sinogram_array)
    nb_angles,size_y,size_x = sinogram_array.shape
    sinogram_tensor = torch.Tensor(sinogram_array)
    filtered_sinogram_array = np.zeros((nb_angles, size_y, size_x))
    # for i in range(size_y):
    #     for angle in range(nb_angles):
    #         filtered_sinogram_array[angle,i,:] = inverse_fft_translate(
    #                                                 ramp_filter(
    #                                                     fft_projs(
    #                                                         sinogram_array[angle,i,:])))

    freq_fft = torch.fft.fftfreq(sinogram_tensor.shape[-2]).reshape((-1, 1))
    ramp = RampFilter()
    ramp = HammingFilter(wl=0.02, wh=0.93)
    filter_total = ramp(freq_fft)

    fig,ax = plt.subplots()
    ax.plot(filter_total.numpy())
    plt.show()

    proj_fft = torch.fft.fft(sinogram_tensor, axis=-2)
    proj_fft = proj_fft * filter_total
    proj_filtered = torch.fft.ifft(proj_fft, axis=-2).real
    filtered_sinogram_array = proj_filtered.numpy()

    filtered_sinogram = itk.image_from_array(filtered_sinogram_array)
    filtered_sinogram.CopyInformation(sinogram)
    itk.imwrite(filtered_sinogram, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
