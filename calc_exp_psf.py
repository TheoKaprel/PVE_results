#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import peak_widths
import pydicom

def calc_fwhm(array, spacing):
    max_profile_x = array.sum(axis=0)
    results_half = peak_widths(max_profile_x,[max_profile_x.argmax()], 0.5)
    fwhm_x = results_half[0][0]*spacing

    max_profile_y = array.sum(axis=1)
    results_half = peak_widths(max_profile_y,[max_profile_y.argmax()], 0.5)
    fwhm_y = results_half[0][0]*spacing

    return fwhm_x,fwhm_y


def main():
    print(args)

    ldistance_1, ldistance_2  = [], []
    lfwhm_x_1, lfwhm_x_2,lfwhm_y_1, lfwhm_y_2=[], [], [], []
    lfwhm_1,lfwhm_2 = [],[]

    for dicom in args.dicomfiles:
        ds = pydicom.dcmread(dicom)
        img = itk.imread(dicom)
        np_array = itk.array_from_image(img)
        spacing, size = img.GetSpacing()[0], np_array.shape[1]
        radial_pos_detect_1 = float(ds[0x54, 0x22][0][0x18, 0x1142].value)
        radial_pos_detect_2 = float(ds[0x54, 0x22][1][0x18, 0x1142].value)

        print(radial_pos_detect_1, radial_pos_detect_2)


        fwhm_x_1,fwhm_y_1 = calc_fwhm(array=np_array[0,:,:],spacing=spacing)
        fwhm_x_2,fwhm_y_2 = calc_fwhm(array=np_array[1,:,:],spacing=spacing)


        lfwhm_x_1.append(fwhm_x_1)
        lfwhm_x_2.append(fwhm_x_2)
        lfwhm_y_1.append(fwhm_y_1)
        lfwhm_y_2.append(fwhm_y_2)

        lfwhm_1.append((fwhm_x_1+fwhm_y_1)/2)
        lfwhm_2.append((fwhm_x_2+fwhm_y_2)/2)

        if 'SOURCE_5CM_DETECTOR' in dicom:
            ldistance_1.append(150)
            ldistance_2.append(50)
        elif 'SOURCE_10_6.NM' in dicom:
            ldistance_1.append(100)
            ldistance_2.append(100)
        elif "15CM" in dicom:
            ldistance_1.append(150)
            ldistance_2.append(150)
        elif "25CM" in dicom:
            ldistance_1.append(250)
            ldistance_2.append(250)
        elif "30CM" in dicom:
            ldistance_1.append(300)
            ldistance_2.append(300)

    ldistance_1=np.array(ldistance_1)
    ldistance_2=np.array(ldistance_2)
    lfwhm_x_1 = np.array(lfwhm_x_1)
    lfwhm_y_1 = np.array(lfwhm_y_1)
    lfwhm_x_2 = np.array(lfwhm_x_2)
    lfwhm_y_2 = np.array(lfwhm_y_2)


    id_1 = np.argsort(ldistance_1)
    id_2 = np.argsort(ldistance_2)

    ldistance_1=ldistance_1[id_1]
    lfwhm_x_1 = lfwhm_x_1[id_1]
    lfwhm_y_1 = lfwhm_y_1[id_1]
    ldistance_2=ldistance_2[id_2]
    lfwhm_x_2 = lfwhm_x_2[id_2]
    lfwhm_y_2 = lfwhm_y_2[id_2]

    print(ldistance_1)

    X = np.concatenate((ldistance_1,ldistance_1,ldistance_2,ldistance_2))
    Y = np.concatenate((lfwhm_x_1,lfwhm_y_1,lfwhm_x_2,lfwhm_y_2))
    n = len(X)

    # linear regression
    # cf : https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Chemometrics_Using_R_(Harvey)/08%3A_Modeling_Data/8.01%3A_Linear_Regression_of_a_Straight-Line_Calibration_Curve
    # and : https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch2slides-simple-linear-regression.pdf

    b1 = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
    b0 = (np.sum(Y) - b1*np.sum(X))/n
    Y_est = b0+b1*X

    s_r = np.sqrt((np.sum((Y - Y_est)**2))/(n-2))
    X_mean=np.mean(X)
    s_b1 = np.sqrt(s_r**2/np.sum((X - X_mean)**2))
    s_b0 = np.sqrt(s_r**2 * np.sum(X**2)/(n*np.sum((X - X_mean)**2)))
    t_95 = 2.78
    print(f"b1 \in [{b1-t_95*s_b1}, {b1+t_95*s_b1}]")
    print(f"b0 \in [{b0-t_95*s_b0}, {b0+t_95*s_b0}]")
    print(f"b1 = {b1}")
    print(f"b0 = {b0}")
    print("donc")
    c = 2 * np.sqrt(2 * np.log(2))
    print(f"alpha_psf = {b1/c}")
    print(f"sigma0_psf = {b0/c}")

    fig,ax = plt.subplots()
    # ax.plot(ldistance_1,lfwhm_1, marker='o', label='det 1')
    # ax.plot(ldistance_2,lfwhm_2, marker='o', label='det 2')
    ax.plot(ldistance_1,lfwhm_x_1, marker="o", label='fwhm_x_1')
    ax.plot(ldistance_1,lfwhm_y_1, marker="o", label='fwhm_y_1')
    ax.plot(ldistance_2,lfwhm_x_2, marker="o", label='fwhm_x_2')
    ax.plot(ldistance_2,lfwhm_y_2, marker="o", label='fwhm_y_2')

    abs = np.linspace(np.min(X), np.max(X), 100)
    Y_est =b0+b1*abs
    Y_b1_max = (b0)+(b1 + t_95 * s_b1)*abs
    Y_b1_min = (b0)+(b1 - t_95 * s_b1)*abs
    ax.plot(abs, Y_est, color='black', label=f'ŷ= {b0} + {b1} x')
    # ax.plot(abs, Y_b1_min, color='black',linestyle=':', label='min b1')
    # ax.plot(abs, Y_b1_max, color='black',linestyle=':', label='max b1')
    # for k in range(100):
    #     b1_rnd=b1 + np.random.randn()*s_b1
    #     ax.plot(abs, b0+b1_rnd*abs, color='black', linestyle='-', alpha=0.5)

    ax.set_xlabel('dist (mm)')
    ax.set_ylabel('fwhm (mm)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicomfiles",  type = str, nargs='*')
    args = parser.parse_args()

    main()
