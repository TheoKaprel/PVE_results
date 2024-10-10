#!/usr/bin/env python3

import argparse
import gatetools
import itk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def main():
    print(args)

    fwhm_array = np.linspace(args.fwhm_min, args.fwhm_max, args.N)
    sigma_array = fwhm_array / (2 * np.sqrt(2 * np.log(2)))
    nrmse_array = np.zeros_like(sigma_array)

    image = itk.imread(args.image)
    image_array = itk.array_from_image(image)

    # normalisation:
    source_img = itk.imread(args.source)
    source_array = itk.array_from_image(source_img)
    sum_src = source_array.sum()
    image_array = image_array*337
    # image_array = image_array / image_array.sum() * sum_src

    for (k,s) in enumerate(sigma_array):
        ls = np.array([s, s, s])
        blurred_source = gatetools.gaussFilter(input = source_img, sigma_mm=ls)
        blurred_source_array = itk.array_from_image(blurred_source)

        nrmse = np.sqrt(np.sum((image_array - blurred_source_array)**2)) / np.sqrt(np.sum(image_array**2))
        nrmse_array[k] = nrmse

    fig,ax = plt.subplots()
    fwhm_array = sigma_array * 2 * np.sqrt(2*np.log(2))
    ax.plot(fwhm_array, nrmse_array)
    ax.set_xlabel("FWHM of filter (mm)")
    ax.set_ylabel("NRMSE")
    plt.show()

    N = args.N

    # fwhm_x = np.linspace(args.fwhm_min, args.fwhm_max, N)
    # fwhm_y = np.linspace(args.fwhm_min, args.fwhm_max, N)
    # fwhm_z = np.linspace(args.fwhm_min, args.fwhm_max, N)
    fwhm_x = np.linspace(8,12, N)
    fwhm_y = np.linspace(15,19, N)
    fwhm_z = np.linspace(9,13, N)
    sigma_x = fwhm_x/ (2 * np.sqrt(2 * np.log(2)))
    sigma_y= fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2)))

    min = 2024
    lsigma = None
    for i in range(N):
        print(i)
        for j in range(N):
            for k in range(N):
                ls = np.array([sigma_x[i], sigma_y[j], sigma_z[k]])
                blurred_source = gatetools.gaussFilter(input=source_img, sigma_mm=ls)
                blurred_source_array = itk.array_from_image(blurred_source)

                nrmse = np.sqrt(np.sum((image_array - blurred_source_array) ** 2)) / np.sqrt(np.sum(image_array ** 2))
                if nrmse < min:
                    lsigma = ls
                    min = nrmse
                    print(ls, nrmse)

    lfwhm = 2*np.sqrt(2*np.log(2)) * lsigma
    print(f"FWHM : {lfwhm}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--image")
    parser.add_argument("--fwhm_min", type=float, default = 0)
    parser.add_argument("--fwhm_max", type=float, default = 13)
    parser.add_argument("-N", type=int, default = 10)
    args = parser.parse_args()

    main()
