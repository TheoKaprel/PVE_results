#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import itk
import pydicom
from scipy.optimize import curve_fit


def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def calc_fwhm(dicomfile, verbose=False):
    ds = pydicom.dcmread(dicomfile)
    radial_pos_detect_1 = float(ds[0x54, 0x22][0][0x18, 0x1142].value)
    radial_pos_detect_2 = float(ds[0x54, 0x22][1][0x18, 0x1142].value)

    img = itk.imread(dicomfile)
    np_array = itk.array_from_image(img)
    spacing,size = img.GetSpacing()[0], np_array.shape[1]
    x = np.linspace(-(spacing*size)/2, spacing*size/2, size)

    proj_det_1 = np_array[0,:,:]
    ind_max = np.unravel_index(proj_det_1.argmax(), proj_det_1.shape)
    max_profile_x = proj_det_1[ind_max[0],:]
    popt_x, pcov_x = curve_fit(gaussian_func, x, max_profile_x)
    fwhm_x_1 = 2*np.sqrt(2*np.log(2))*(popt_x[2])
    max_profile_y = proj_det_1[:,ind_max[1]]
    popt_y, pcov_y = curve_fit(gaussian_func, x, max_profile_y)
    fwhm_y_1 = 2*np.sqrt(2*np.log(2))*(popt_y[2])

    proj_det_2 = np_array[1,:,:]
    ind_max = np.unravel_index(proj_det_2.argmax(), proj_det_2.shape)
    max_profile_x = proj_det_2[ind_max[0],:]
    popt_x, pcov_x = curve_fit(gaussian_func, x, max_profile_x)
    fwhm_x_2 = 2*np.sqrt(2*np.log(2))*(popt_x[2])
    max_profile_y = proj_det_2[:,ind_max[1]]
    popt_y, pcov_y = curve_fit(gaussian_func, x, max_profile_y)
    fwhm_y_2 = 2*np.sqrt(2*np.log(2))*(popt_y[2])

    fwhm_x = [fwhm_x_1,fwhm_x_2]
    fwhm_y = [fwhm_y_1,fwhm_y_2]
    radial_pos = [radial_pos_detect_1, radial_pos_detect_2]

    if verbose:
        for fx,fy,r in zip(fwhm_x, fwhm_y, radial_pos):
            print(f'(rad = {r}) FWHM_x={fx} / FWHM_y={fy}')

    return fwhm_x,fwhm_y,radial_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dicomfile")
    args = parser.parse_args()

    _,__,___ = calc_fwhm(dicomfile=args.dicomfile, verbose = True)
