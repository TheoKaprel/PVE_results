#!/usr/bin/env python3

import argparse
from calc_fwhm import calc_fwhm,calc_fwhm2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    print(args)

    fwhm_simu1=[]
    c = 2*np.sqrt(2*np.log(2))

    dist_simu_mm = [d__*10 for d__ in args.distsimu]
    for dist in args.distsimu:
        simu_file=args.simufiles1.replace('%d', str(int(dist)))
        fwhm_x,fwhm_y,_ = calc_fwhm2(projectionfile=simu_file)
        print(f'fx {fwhm_x} / fy {fwhm_y}')
        fwhm_simu1.append(np.mean(fwhm_x + fwhm_y))

    regression_model = LinearRegression()
    X = np.array(dist_simu_mm).reshape(-1, 1)
    y = np.array(fwhm_simu1).reshape(-1, 1)
    regression_model.fit(X, y)
    print('*'*30)
    print('simu 1')
    pente_fwhm=regression_model.coef_[0][0]
    ord_orig_fwhm=regression_model.intercept_[0]
    print(f"linear reg (sigma) : pente = {pente_fwhm/c} / orig = {ord_orig_fwhm/c}")


    fwhm_simu2=[]
    for dist in args.distsimu:
        simu_file=args.simufiles2.replace('%d', str(int(dist)))
        fwhm_x,fwhm_y,_ = calc_fwhm2(projectionfile=simu_file)
        print(f'fx {fwhm_x} / fy {fwhm_y}')
        fwhm_simu2.append(np.mean(fwhm_x+fwhm_y))

    regression_model = LinearRegression()
    X = np.array(dist_simu_mm).reshape(-1, 1)
    y = np.array(fwhm_simu2).reshape(-1, 1)
    regression_model.fit(X, y)
    print('*'*30)
    print('simu 2')
    pente_fwhm=regression_model.coef_[0][0]
    ord_orig_fwhm=regression_model.intercept_[0]
    print(f"linear ref : pente = {pente_fwhm/c} / orig = {ord_orig_fwhm/c}")



    fwhm_simu3=[]
    for dist in args.distsimu:
        simu_file=args.simufiles3.replace('%d', str(int(dist)))
        fwhm_x,fwhm_y,_ = calc_fwhm2(projectionfile=simu_file)
        print(f'fx {fwhm_x} / fy {fwhm_y}')
        fwhm_simu3.append(np.mean(fwhm_x+fwhm_y))

    regression_model = LinearRegression()
    X = np.array(dist_simu_mm).reshape(-1, 1)
    y = np.array(fwhm_simu3).reshape(-1, 1)
    regression_model.fit(X, y)
    print('*'*30)
    print('simu 3')
    pente_fwhm=regression_model.coef_[0][0]
    ord_orig_fwhm=regression_model.intercept_[0]
    print(f"linear ref : pente = {pente_fwhm/c} / orig = {ord_orig_fwhm/c}")



    # dist_exp_mm, fwhm_exp,fwhm_exp2 = [],[], []
    # for dicom in args.dicomfiles:
    #     fwhm_x,fwhm_y,radial_pos = calc_fwhm2(projectionfile=dicom)
    #     if 'SOURCE_5CM_DETECTOR' in dicom:
    #         dist_exp_mm.append(50)
    #     else:
    #         dist_exp_mm.append(radial_pos[1])
    #     fwhm_exp.append((fwhm_x[1]+fwhm_y[1])/2)

    # lRint = np.arange(1, 5.5, step = 0.005)
    # regression_model = LinearRegression()
    # fig_R,ax_R = plt.subplots()
    # lR2 = []
    # for rint in lRint:
    #     Rcoll = [np.sqrt(rsys**2 - rint**2) for rsys in fwhm_exp]
    #     X= np.array(dist_exp_mm).reshape(-1, 1)
    #     y = np.array(Rcoll).reshape(-1, 1)
    #     regression_model.fit(X, y)
    #     lR2.append(regression_model.score(X,y))
    # ax_R.plot(lRint, lR2)
    # Rint = lRint[np.argmax(lR2)]
    # ax_R.axvline(Rint, linestyle = 'dashed')
    # ax_R.set_xlabel('intrisic resolution', fontsize=18)
    # ax_R.set_ylabel('Linear Regression Score (R^2)', fontsize=18)
    # plt.xticks(list(plt.xticks()[0]) + [Rint])
    # # plt.show()

    # Rcoll = [np.sqrt(rsys ** 2 - Rint ** 2) for rsys in fwhm_exp]
    # X = np.array(dist_exp_mm).reshape(-1, 1)
    # y = np.array(Rcoll).reshape(-1, 1)
    # regression_model.fit(X, y)
    # print('*'*30)
    # print('EXP DATA without intrinsic resol : ')
    # pente_fwhm=regression_model.coef_[0][0]
    # ord_orig_fwhm=regression_model.intercept_[0]
    # c = (2*np.sqrt(2*np.log(2)))

    # print(f"FWHM(d) = {pente_fwhm} d + {ord_orig_fwhm}")
    # print(f"sigma(d) = {pente_fwhm/c} d + {ord_orig_fwhm/c}")
    # print("Intrisic Resolution : ", Rint)

    # X = np.array(dist_exp_mm).reshape(-1, 1)
    # y = np.array(fwhm_exp).reshape(-1, 1)
    # regression_model.fit(X, y)
    # print('*'*30)
    # print('EXP DATA including intrinsic resol : ')
    # pente_fwhm_=regression_model.coef_[0][0]
    # ord_orig_fwhm_=regression_model.intercept_[0]
    #
    # print(f"FWHM(d) = {pente_fwhm_} d + {ord_orig_fwhm_}")
    # print(f"sigma(d) = {pente_fwhm_/c} d + {ord_orig_fwhm_/c}")


    # print('*'*30)
    # print('SIMU DATA without intrinsic resol : ')
    # X = np.array(dist_simu_mm).reshape(-1, 1)
    # y = np.array(fwhm_simu1).reshape(-1, 1)
    # regression_model.fit(X, y)
    # pente_fwhm__=regression_model.coef_[0][0]
    # ord_orig_fwhm__=regression_model.intercept_[0]

    # print(f"FWHM(d) = {pente_fwhm__} d + {ord_orig_fwhm__}")
    # print(f"sigma(d) = {pente_fwhm__/c} d + {ord_orig_fwhm__/c}")


    fig,ax = plt.subplots()
    ax.set_xlabel('PTSRC to Detector distance (mm)', fontsize=18)
    ax.set_ylabel('FWHM (mm)', fontsize=18)
    ax.plot(dist_simu_mm,fwhm_simu1, color='yellow', label=args.labels[0], marker='o')
    ax.plot(dist_simu_mm,fwhm_simu2, color='gold', label=args.labels[1], marker='o')
    ax.plot(dist_simu_mm,fwhm_simu3, color='darkorange', label=args.labels[2], marker='o')
    # ax.plot(dist_exp_mm, fwhm_exp, color="darkblue", label="fwhm exp", marker='o')
    # ax.plot(dist_exp_mm, Rcoll, color="royalblue", label="fwhm-colli exp", marker='o')
    # ax.plot(dist_exp_mm, [pente_fwhm_*d + ord_orig_fwhm_ for d in dist_exp_mm], color="darkblue",alpha=0.5, label="fit exp with intrinsic resol")
    # ax.plot(dist_exp_mm, [pente_fwhm*d + ord_orig_fwhm for d in dist_exp_mm], color="royalblue",alpha=0.5, label="fit exp without intrinsic resol")
    # ax.plot(dist_simu_mm, [pente_fwhm__*d + ord_orig_fwhm__ for d in dist_simu_mm], color="orange",alpha=0.5, label="fit simu without intrinsic resol")
    plt.legend(fontsize=18)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simufiles1", type = str)
    parser.add_argument("--simufiles2", type = str)
    parser.add_argument("--simufiles3", type = str)
    parser.add_argument("--distsimu", type = float, nargs='*')
    parser.add_argument("--dicomfiles", type = str, nargs='*')
    parser.add_argument("--labels", type = str, nargs='*')
    args = parser.parse_args()

    main()
