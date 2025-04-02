#!/usr/bin/env python3

import argparse
import itk
from itk import RTK as rtk
import numpy as np
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

def main():
    print(args)
    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]


    projs = itk.imread(args.projs, pixelType)
    attmap = itk.imread(args.attmap, pixelType)

    nprojs = args.nprojs
    sid = args.sid

    Offset = projs.GetOrigin()
    list_angles = np.linspace(0, 360, nprojs + 1)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(nprojs):
        geometry.AddProjection(sid, 0, list_angles[i], Offset[0], Offset[1])

    # filter projs
    projections_torch = torch.from_numpy(itk.array_from_image(projs))
    # axis = 2
    # freq_fft = torch.fft.fftfreq(projections_torch.shape[axis]).reshape((-1, 1))
    # projs_filter = RampFilter
    # filter_total = projs_filter()(freq_fft)
    # proj_fft = torch.fft.fft(projections_torch, axis=axis)
    # proj_fft = proj_fft * filter_total
    # proj_filtered = torch.fft.ifft(proj_fft, axis=axis).real


    freq_fft = torch.fft.fftfreq(projections_torch.shape[-2])
    filtr1 = HammingFilter(wl=0.5, wh=1)
    filtr2 = RampFilter()
    filter_total = filtr1(freq_fft)*filtr2(freq_fft)
    # filter_total = filtr1(freq_fft)
    print(projections_torch.shape)
    print(filter_total.shape)
    proj_filtered = torch.zeros_like(projections_torch)
    for i in range(128):
        for angle in range(120):
            proj_fft = torch.fft.fft(projections_torch[angle,i,:])
            proj_fft = proj_fft * filter_total
            proj_filtered[angle,i,:] = torch.fft.ifft(proj_fft).real

    proj_filtered_itk = itk.image_from_array(proj_filtered.numpy())
    proj_filtered_itk.CopyInformation(projs)

    # output image info
    like_image = itk.imread(args.like, pixelType)
    constant_image = rtk.ConstantImageSource[imageType].New()
    constant_image.SetSpacing(like_image.GetSpacing())
    constant_image.SetOrigin(like_image.GetOrigin())
    constant_image.SetSize(itk.size(like_image))
    constant_image.SetConstant(1)
    output_image = constant_image.GetOutput()

    backprojs_filter = rtk.ZengBackProjectionImageFilter.New()
    backprojs_filter.SetGeometry(geometry)
    backprojs_filter.SetInput(0, output_image)
    backprojs_filter.SetInput(1, proj_filtered_itk)
    backprojs_filter.SetInput(2, attmap)
    backprojs_filter.SetSigmaZero(0)
    backprojs_filter.SetAlpha(0)

    backprojs_filter.Update()
    output_backproj = backprojs_filter.GetOutput()
    itk.imwrite(output_backproj, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--attmap")
    parser.add_argument("--nprojs", type = int)
    parser.add_argument("--sid", type = float)
    parser.add_argument("--like")
    parser.add_argument("--output")

    args = parser.parse_args()

    main()
