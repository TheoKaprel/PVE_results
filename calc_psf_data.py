#!/usr/bin/env python3

import argparse
from calc_fwhm import calc_fwhm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    print(args)

    ldistance_1, ldistance_2,lfwhm_1,lfwhm_2  = [], [], [], []

    # mean_dist, mean_fwhm = [],[]
    fwhm_simu=[]

    for dicom in args.dicomfiles:
        fwhm_x,fwhm_y,radial_pos = calc_fwhm(projectionfile=dicom)
        if 'SOURCE_5CM_DETECTOR' in dicom:
            # ldistance_1.append(162)
            ldistance_2.append(50)
            # lfwhm_1.append((fwhm_y[0]+fwhm_x[0])/2)
            lfwhm_2.append((fwhm_y[1]+fwhm_x[1])/2)
            # mean_dist.append(50)
            # mean_fwhm.append((fwhm_y[1]+fwhm_x[1])/2)
        else:
            ldistance_1.append(radial_pos[0])
            ldistance_2.append(radial_pos[1])
            lfwhm_1.append((fwhm_x[0]+fwhm_y[0])/2)
            lfwhm_2.append((fwhm_x[1]+fwhm_y[1])/2)

            # mean_dist.append(np.mean(radial_pos))
            # mean_fwhm.append((np.mean(fwhm_x+fwhm_y)))

    # mean_dist = ldistance_2
    # mean_fwhm = lfwhm_2

    dist_simu = args.distsimu
    for simu in args.simufiles:
        fwhm_x,fwhm_y,_ = calc_fwhm(projectionfile=simu)
        print(f'fx {fwhm_x} / fy {fwhm_y}')
        fwhm_simu.append(np.mean(fwhm_x+fwhm_y))

    lRint = np.arange(1, 5, step = 0.05)
    regression_model = LinearRegression()
    fig_R,ax_R = plt.subplots()
    lR2 = []
    for rint in lRint:
        Rcoll = [np.sqrt(rsys**2 - rint**2) for rsys in lfwhm_2]
        X= np.array(ldistance_2).reshape(-1, 1)
        y = np.array(Rcoll).reshape(-1, 1)
        regression_model.fit(X, y)
        lR2.append(regression_model.score(X,y))
    ax_R.plot(lRint, lR2)
    Rint = lRint[np.argmax(lR2)]
    ax_R.axvline(Rint, linestyle = 'dashed')
    ax_R.set_xlabel('intrisic resolution', fontsize=18)
    ax_R.set_ylabel('Linear Regression Score (R^2)', fontsize=18)
    plt.xticks(list(plt.xticks()[0]) + [Rint])
    plt.show()

    # Rint = 3.91024 # from siemens specifications

    Rcoll = [np.sqrt(rsys ** 2 - Rint ** 2) for rsys in lfwhm_2]
    X = np.array(ldistance_2).reshape(-1, 1)
    y = np.array(Rcoll).reshape(-1, 1)
    regression_model.fit(X, y)
    print('*'*30)
    print('EXP DATA : ')
    print("Pente : ", regression_model.coef_[0])
    print("Ordonnée à l'origine : ", regression_model.intercept_)
    print("Intrisic Resolution : ", Rint)

    X = np.array(dist_simu).reshape(-1, 1)
    y = np.array(fwhm_simu).reshape(-1, 1)
    regression_model_rtk = LinearRegression()
    regression_model_rtk.fit(X, y)
    print('*'*30)
    print('RTK DATA : ')
    print("Pente : ", regression_model_rtk.coef_[0])
    print("Ordonnée à l'origine : ", regression_model_rtk.intercept_)


    fig,ax = plt.subplots()
    # ax.plot(ldistance_1, lfwhm_1, color = 'blue',label = 'd1', marker = 'o', markersize=5, linewidth = 1.5)
    # ax.plot(ldistance_2, lfwhm_2, color = 'green',label = 'd2', marker = 'o', markersize=5, linewidth = 1.5)
    # ax.plot(mean_dist, mean_fwhm, color = 'black',label = 'mean', marker = 'o', markersize=5, linewidth = 1.5)

    ax.set_xlabel('PTSRC to Detector distance (mm)', fontsize=18)
    ax.set_ylabel('FWHM (mm)', fontsize=18)


    d = np.arange(start=0, stop=350, step = 10)

    Rcoll__ = regression_model.coef_[0]*d + regression_model.intercept_
    Rsys__ = np.sqrt(Rint**2 + Rcoll__**2)
    ax.scatter(ldistance_2, lfwhm_2, color = 'black', label = "Exp data (Rsystem)", s = 40)
    ax.scatter(ldistance_2, Rcoll, color = 'grey', label = "Exp data (Rcoll)", s = 40)
    ax.plot(d, Rcoll__, label = "Rcoll", linestyle = 'dashed', color = 'grey', linewidth = 2)
    ax.plot(d, Rsys__, label = f"Rsys (Rint={round(Rint,3)})", color = 'black', linewidth = 2, linestyle= 'dashed')

    ax.plot(dist_simu, fwhm_simu, color = 'orange',label = 'simu', marker = 'o', markersize=7, linewidth = 2)
    plt.legend(fontsize=18)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicomfiles", type = str, nargs='*')
    parser.add_argument("--simufiles", type = str, nargs='*')
    parser.add_argument("--distsimu", type = float, nargs='*')
    args = parser.parse_args()

    main()
