#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from itk import RTK as rtk
from sklearn.linear_model import LinearRegression

import itk
from scipy.signal import peak_widths


def calc_fwhm(array, spacing):
    max_profile_x = array.sum(axis=0)
    results_half = peak_widths(max_profile_x,[max_profile_x.argmax()], 0.5)
    fwhm_x = results_half[0][0]*spacing

    max_profile_y = array.sum(axis=1)
    results_half = peak_widths(max_profile_y,[max_profile_y.argmax()], 0.5)
    fwhm_y = results_half[0][0]*spacing

    return fwhm_x, fwhm_y


def linear_regr(x,y):
    regression_model = LinearRegression()
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y).reshape(-1, 1)
    regression_model.fit(X, Y)
    pente = regression_model.coef_[0][0]
    ordo = regression_model.intercept_[0]
    return pente, ordo


def main():
    print(args)

    fig, ax = plt.subplots()

    if args.real is not None:
        ldistance_1, ldistance_2  = [], []
        lfwhm_x_1, lfwhm_x_2,lfwhm_y_1, lfwhm_y_2=[], [], [], []
        lfwhm_1,lfwhm_2 = [],[]

        for dicom in args.real:
            img = itk.imread(dicom)
            np_array = itk.array_from_image(img)
            spacing, size = img.GetSpacing()[0], np_array.shape[1]

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




        ax.plot(ldistance_1,lfwhm_x_1, marker="o", label='real exp Tc99m-LEHR', color='green')
        ax.plot(ldistance_1,lfwhm_y_1, marker="o", color='green')
        ax.plot(ldistance_2,lfwhm_x_2, marker="o", color='green')
        ax.plot(ldistance_2,lfwhm_y_2, marker="o", color='green')

    dists = [100, 150, 200, 250, 300]
    if args.simu1 is not None:
        lfwhm_x,lfwhm_y=[],[]
        for fnames in args.simu1:
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[4, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)
        ax.plot(dists,lfwhm_x, marker='o', label='simu1 fwhm_x', color = 'blue')
        ax.plot(dists,lfwhm_y, marker='s', label='simu1 fwhm_y', color = 'blue')

    if args.simu2 is not None:
        lfwhm_x,lfwhm_y=[],[]
        for fnames in args.simu2:
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[4, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)
        ax.plot(dists,lfwhm_x, marker='o', label='simu Lu177-MEGP', color = 'orange')
        ax.plot(dists,lfwhm_y, marker='s', color = 'orange')

    if args.simu3 is not None:
        lfwhm_x,lfwhm_y=[],[]
        dists=[]
        for fnames in args.simu3:
            i = fnames.find('mm')
            dists.append(float(fnames[i-3:i]))
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[0, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)
        ax.plot(dists,lfwhm_x, marker='o', label='simu3 fwhm_x', color = 'blueviolet')
        ax.plot(dists,lfwhm_y, marker='s', label='simu3 fwhm_y', color = 'blueviolet')

    if args.simu4 is not None:
        lfwhm_x,lfwhm_y=[],[]
        dists=[]
        for fnames in args.simu4:
            i = fnames.find('mm')
            dists.append(float(fnames[i-3:i]))
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[4, :, :], spacing=2.3976*2)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)
        ax.plot(dists,lfwhm_x, marker='o', label='simu4 fwhm_x', color = 'pink')
        ax.plot(dists,lfwhm_y, marker='s', label='simu4 fwhm_y', color = 'pink')

    if args.simu5 is not None:
        lfwhm_x,lfwhm_y=[],[]
        dists=[]
        for fnames in args.simu5:
            i = fnames.find('mm')
            dists.append(float(fnames[i-3:i]))
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[4, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)
        ax.plot(dists,lfwhm_x, marker='o', label='simu5 fwhm_x', color = 'black')
        ax.plot(dists,lfwhm_y, marker='s', label='simu5 fwhm_y', color = 'black')

        lfwhm = [(fx+fy)/2 for (fx,fy) in zip(lfwhm_x,lfwhm_y)]
        alpha_fwhm, fwhm_0 = linear_regr(dists,lfwhm)
        ax.plot(dists, alpha_fwhm*np.array(dists) + fwhm_0, linestyle="dashed", color = 'black', label=f'simu5 lin.regr.')
        print(f"Simu5 Linear Regression")
        print(f"alpha_fwhm : {alpha_fwhm}")
        print(f"fwhm0 : {fwhm_0}")
        c = 2 * np.sqrt(2 * np.log(2))
        print(f"alpha : {alpha_fwhm/c}")
        print(f"sigma0 : {fwhm_0/c}")

    ax.set_ylim([0, 30])
    ax.set_xlabel("Source to detector distance (mm)",fontsize=18)
    ax.set_ylabel("FWHM", fontsize=18)
    ax.legend(fontsize=12)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", nargs='*')
    parser.add_argument("--simu1", nargs='*')
    parser.add_argument("--simu2", nargs="*")
    parser.add_argument("--simu3", nargs="*")
    parser.add_argument("--simu4", nargs="*")
    parser.add_argument("--simu5", nargs="*")
    args = parser.parse_args()

    main()
