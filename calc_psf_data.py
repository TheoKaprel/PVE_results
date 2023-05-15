#!/usr/bin/env python3

import argparse
from calc_fwhm import calc_fwhm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    print(args)

    ldistance_1, ldistance_2,lfwhm_x_1, lfwhm_x_2, lfwhm_y_1, lfwhm_y_2 = [],[], [], [], [], []

    mean_dist, mean_fwhm = [],[]

    for dicom in args.dicomfiles:
        fwhm_x,fwhm_y,radial_pos = calc_fwhm(dicomfile=dicom,verbose=False)
        ldistance_1.append(radial_pos[0])
        ldistance_2.append(radial_pos[1])
        lfwhm_x_1.append(fwhm_x[0])
        lfwhm_x_2.append(fwhm_x[1])
        lfwhm_y_1.append(fwhm_y[0])
        lfwhm_y_2.append(fwhm_y[1])


        mean_dist.append(np.mean(radial_pos))
        mean_fwhm.append((np.mean(fwhm_x+fwhm_y)))

    print(mean_dist)
    print(mean_fwhm)

    regression_model = LinearRegression()
    regression_model.fit(np.array(mean_dist).reshape(-1, 1), np.array(mean_fwhm).reshape(-1, 1))
    print("Pente : ", regression_model.coef_[0])
    print("Ordonnée à l'origine : ", regression_model.intercept_)


    fig,ax = plt.subplots()
    ax.plot(ldistance_1, lfwhm_x_1, color = 'blue',label = 'd1_x', marker = 'o', markersize=5, linewidth = 1.5)
    ax.plot(ldistance_1, lfwhm_y_1, color = 'green',label = 'd1_y', marker = 'o', markersize=5, linewidth = 1.5)
    ax.plot(ldistance_2, lfwhm_x_2, color = 'orange',label = 'd2_x', marker = 'o', markersize=5, linewidth = 1.5)
    ax.plot(ldistance_2, lfwhm_y_2, color = 'red',label = 'd2_y', marker = 'o', markersize=5, linewidth = 1.5)
    ax.plot(mean_dist, mean_fwhm, color = 'black',label = 'mean', marker = 'o', markersize=5, linewidth = 1.5)
    ax.set_xlabel('PTSRC to Detector distance (mm)')
    ax.set_ylabel('FWHM (mm)')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dicomfiles", type = str, nargs='*')
    args = parser.parse_args()

    main()
