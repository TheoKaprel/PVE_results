#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import itk
import pydicom
from scipy.optimize import curve_fit
from scipy.signal import peak_widths

def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def calc_fwhm(projectionfile, verbose=False):

    if projectionfile[-3:]=='IMA':
        ds = pydicom.dcmread(projectionfile)
        radial_pos_detect_1 = float(ds[0x54, 0x22][0][0x18, 0x1142].value)
        radial_pos_detect_2 = float(ds[0x54, 0x22][1][0x18, 0x1142].value)
        radial_pos = [radial_pos_detect_1, radial_pos_detect_2]
        is_dicom=True
    else:
        radial_pos = None
        is_dicom=False

    img = itk.imread(projectionfile)
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

    if is_dicom:
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
    else:
        fwhm_x = [fwhm_x_1]
        fwhm_y = [fwhm_y_1]

    return fwhm_x,fwhm_y,radial_pos


def calc_fwhm2(projectionfile, verbose=False):

    if projectionfile[-3:]=='IMA':
        ds = pydicom.dcmread(projectionfile)
        radial_pos_detect_1 = float(ds[0x54, 0x22][0][0x18, 0x1142].value)
        radial_pos_detect_2 = float(ds[0x54, 0x22][1][0x18, 0x1142].value)
        radial_pos = [radial_pos_detect_1, radial_pos_detect_2]
        is_dicom=True
    else:
        radial_pos = None
        is_dicom=False

    img = itk.imread(projectionfile)
    np_array = itk.array_from_image(img)
    spacing,size = img.GetSpacing()[0], np_array.shape[1]
    # x = np.linspace(-(spacing*size)/2, spacing*size/2, size)

    proj_det_1 = np_array[0,:,:]
    max_profile_x = proj_det_1.sum(axis=0)
    results_half = peak_widths(max_profile_x,[max_profile_x.argmax()], 0.5)
    fwhm_x_1 = results_half[0][0]*spacing

    # plt.plot(max_profile_x, marker='o')
    # plt.plot([max_profile_x.argmax()], max_profile_x[max_profile_x.argmax()], "x")
    # plt.hlines(*results_half[1:], color="C2")
    # print(results_half[0], fwhm_x_1)
    # print(max_profile_x.argmax())
    # sigma=results_half[0][0]/(2*np.sqrt(2*np.log(2)))
    # x0=max_profile_x.argmax()
    # a=max_profile_x[x0]
    # gauss_fit=gaussian_func(x=np.linspace(0, 256, 1024),a=a,x0=x0,sigma=sigma)
    # plt.plot(np.linspace(0, 256, 1024),gauss_fit, color="black", alpha=0.8)
    # plt.show()

    max_profile_y = proj_det_1.sum(axis=1)
    results_half = peak_widths(max_profile_y,[max_profile_y.argmax()], 0.5)
    fwhm_y_1 = results_half[0][0]*spacing

    # plt.plot(max_profile_y)
    # plt.plot([ind_max[1]], max_profile_y[ind_max[1]], "x")
    # print(results_half[0])
    # plt.hlines(*results_half[1:], color="C2")
    # plt.show()

    if is_dicom:
        proj_det_2 = np_array[1,:,:]
        # ind_max = np.unravel_index(proj_det_2.argmax(), proj_det_2.shape)
        max_profile_x = proj_det_2.sum(axis=0)
        results_half = peak_widths(max_profile_x, [max_profile_x.argmax()], 0.5)
        fwhm_x_2 = results_half[0][0] * spacing

        max_profile_y = proj_det_2.sum(axis=1)
        results_half = peak_widths(max_profile_y, [max_profile_y.argmax()], 0.5)
        fwhm_y_2 = results_half[0][0] * spacing
        fwhm_x = [fwhm_x_1,fwhm_x_2]
        fwhm_y = [fwhm_y_1,fwhm_y_2]

        print(fwhm_x_2,fwhm_y_2)

    else:
        fwhm_x = [fwhm_x_1]
        fwhm_y = [fwhm_y_1]

    return fwhm_x,fwhm_y,radial_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("projectionfile")
    args = parser.parse_args()

    _,__,___ = calc_fwhm(projectionfile=args.projectionfile)
