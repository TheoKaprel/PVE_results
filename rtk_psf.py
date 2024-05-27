#!/usr/bin/env python3

import argparse
from itk import RTK as rtk
import itk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import peak_widths
import matplotlib.cm as cm



def calc_fwhm(array, spacing):
    max_profile_x = array.sum(axis=0)
    results_half = peak_widths(max_profile_x,[max_profile_x.argmax()], 0.5)
    fwhm_x = results_half[0][0]*spacing

    max_profile_y = array.sum(axis=1)
    results_half = peak_widths(max_profile_y,[max_profile_y.argmax()], 0.5)
    fwhm_y = results_half[0][0]*spacing

    return (fwhm_x+fwhm_y)/2


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

    if args.load==False:
        pixelType = itk.F
        imageType = itk.Image[pixelType, 3]
        source=itk.imread(args.source, pixelType)
        dists=[100, 150, 200, 250, 300, 350]

        proj_size=[256, 256, len(dists)]
        proj_spacing=[2.3976, 2.3976, 1]
        proj_offset= [-(s-1)*sp/2 for s,sp in zip(proj_size,proj_spacing)]

        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for d in dists:
            geometry.AddProjection(d, 0, 0, proj_offset[0], proj_offset[1])

        c = 2*np.sqrt(2*np.log(2))

        output_proj = rtk.ConstantImageSource[imageType].New()
        output_proj.SetSpacing(proj_spacing)
        output_proj.SetOrigin(proj_offset)
        output_proj.SetSize(proj_size)
        output_proj.SetConstant(0.)

        forward_projector = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector.SetInput(0, output_proj.GetOutput())
        forward_projector.SetInput(1, source)
        forward_projector.SetGeometry(geometry)

    if args.sigma0:
        fig_sigma0, ax_sigma0 = plt.subplots()

        # sigma
        lsigma = np.linspace(0, 3, 20)
        for alpha in [0.018, 0.020]:
            l_est_sigma=[]
            for sig in lsigma:
                forward_projector.SetSigmaZero(sig)
                forward_projector.SetAlpha(alpha)
                forward_projector.Update()
                output_proj = forward_projector.GetOutput()
                output_proj_array=itk.array_from_image(output_proj)

                lfwhm=[]
                for p in range(output_proj_array.shape[0]):
                    fwhm= calc_fwhm(array=output_proj_array[p,:,:], spacing=proj_spacing[0])
                    lfwhm.append(fwhm)

                regression_model = LinearRegression()
                X = np.array(dists).reshape(-1, 1)
                y = np.array(lfwhm).reshape(-1, 1)
                regression_model.fit(X, y)
                pente_fwhm=regression_model.coef_[0][0]
                ord_orig_fwhm=regression_model.intercept_[0]
                l_est_sigma.append(ord_orig_fwhm/c)

            ax_sigma0.plot(lsigma, l_est_sigma, marker='o', label=f'alpha = {alpha}')

        ax_sigma0.plot(lsigma, lsigma, linestyle=':')


        ax_sigma0.set_xlabel('sigma0 rtk')
        ax_sigma0.set_ylabel('sigma0 estimated (lin regr)')
        ax_sigma0.legend()

    if args.alpha:
        fig, ax_alpha = plt.subplots()
        # sigma
        lalpha = np.linspace(0.016, 0.019, 3)
        for sigma0 in [1,2,3]:
            l_est_alpha=[]
            for alph in lalpha:
                forward_projector.SetSigmaZero(sigma0)
                forward_projector.SetAlpha(alph)
                forward_projector.Update()
                output_proj = forward_projector.GetOutput()
                output_proj_array=itk.array_from_image(output_proj)

                lfwhm=[]
                for p in range(output_proj_array.shape[0]):
                    fwhm= calc_fwhm(array=output_proj_array[p,:,:], spacing=proj_spacing[0])
                    lfwhm.append(fwhm)

                regression_model = LinearRegression()
                X = np.array(dists).reshape(-1, 1)
                y = np.array(lfwhm).reshape(-1, 1)
                regression_model.fit(X, y)
                pente_fwhm=regression_model.coef_[0][0]
                ord_orig_fwhm=regression_model.intercept_[0]
                l_est_alpha.append(pente_fwhm/c)

            ax_alpha.plot(lalpha, l_est_alpha, marker='o', label=f'sigma0 = {sigma0}')

        ax_alpha.plot(lalpha, lalpha, linestyle=':')
        ax_alpha.set_xlabel('alpha rtk')
        ax_alpha.set_ylabel('alpha estimated (lin regr)')
        ax_alpha.legend()

    if args.both:
        # exp_b1, exp_b0 = 0.04298686361940656, 3.7235381602325006
        # exp_fit = exp_b0 + np.array(dists) * exp_b1

        fig, ax_both = plt.subplots(2,2)

        # sigma
        lalpha = np.linspace(0.0173, 0.0177, 10)
        lsigma = np.linspace(1.85, 2, 10)
        est_alpha=np.zeros((len(lsigma), len(lalpha)))
        est_sigma=np.zeros((len(lsigma), len(lalpha)))
        for i,sigma0 in enumerate(lsigma):
            print(i)
            for j,alph in enumerate(lalpha):

                forward_projector.SetSigmaZero(sigma0)
                forward_projector.SetAlpha(alph)
                forward_projector.Update()
                output_proj = forward_projector.GetOutput()
                output_proj_array=itk.array_from_image(output_proj)

                lfwhm_rtk=[]
                for p in range(output_proj_array.shape[0]):
                    fwhm= calc_fwhm(array=output_proj_array[p,:,:], spacing=proj_spacing[0])
                    lfwhm_rtk.append(fwhm)

                regression_model = LinearRegression()
                X = np.array(dists).reshape(-1, 1)
                y = np.array(lfwhm_rtk).reshape(-1, 1)
                regression_model.fit(X, y)
                pente_fwhm = regression_model.coef_[0][0]
                ord_orig_fwhm = regression_model.intercept_[0]

                est_alpha[i,j] = pente_fwhm / c
                est_sigma[i,j] = ord_orig_fwhm / c

        np.save('lalpha.npy', lalpha)
        np.save('lsigma.npy', lsigma)
        np.save('est_alpha.npy', est_alpha)
        np.save('est_sigma.npy', est_sigma)

        for k in range(len(lsigma)):
            ax_both[0,0].plot(lalpha, est_alpha[k,:], color=cm.jet(k/len(lsigma)))
            ax_both[1,0].plot(lalpha, est_sigma[k,:], color=cm.jet(k/len(lsigma)))
        ax_both[0,0].set_xlabel('alpha')
        ax_both[1,0].set_xlabel('alpha')
        ax_both[0,0].set_ylabel('alpha_est')
        ax_both[1,0].set_ylabel('sigma_est')

        for k in range(len(lalpha)):
            ax_both[0,1].plot(lsigma, est_alpha[:,k], color=cm.jet(k/len(lsigma)))
            ax_both[1,1].plot(lsigma, est_sigma[:,k], color=cm.jet(k/len(lsigma)))
        ax_both[0,1].set_xlabel('sigma')
        ax_both[1,1].set_xlabel('sigma')
        ax_both[0,1].set_ylabel('alpha_est')
        ax_both[1,1].set_ylabel('sigma_est')


    if args.load:
        exp_alpha=0.018254840198984965
        exp_sigma0 = 1.5812410668449028
        dist=np.linspace(0, 300, 100)
        exp_y = exp_alpha * dist + exp_sigma0

        lalpha = np.load('lalpha.npy')
        lsigma = np.load('lsigma.npy')
        est_alpha = np.load('est_alpha.npy')
        est_sigma = np.load('est_sigma.npy')
        print("lalpha : ", lalpha)
        print("lsigma : ", lsigma)

        fig,ax_err=plt.subplots()
        err = np.zeros_like(est_sigma)
        for i in range(est_sigma.shape[0]):
            for j in range(est_sigma.shape[1]):
                alpha = est_alpha[i,j]
                sigma = est_sigma[i,j]
                est_y = alpha * dist + sigma
                err[i,j] = np.mean((est_y - exp_y)**2)
        ax_err.imshow(err)
        ax_err.set_title('err')
        ind = np.unravel_index(np.argmin(err, axis=None), err.shape)
        print(err[ind])
        print("alpha ",est_alpha[ind])
        print("sigma ",est_sigma[ind])

        print("alpha_rtk : ", lalpha[ind[1]])
        print("sigma_rtk : ", lsigma[ind[0]])

        fig, ax_both = plt.subplots(2, 2)
        for k in range(len(lsigma)):
            ax_both[0, 0].plot(lalpha, est_alpha[k, :], color=cm.jet(k / len(lsigma)))
            ax_both[1, 0].plot(lalpha, est_sigma[k, :], color=cm.jet(k / len(lsigma)))
        ax_both[0, 0].set_xlabel('alpha')
        ax_both[1, 0].set_xlabel('alpha')
        ax_both[0, 0].set_ylabel('alpha_est')
        ax_both[1, 0].set_ylabel('sigma_est')

        for k in range(len(lalpha)):
            ax_both[0, 1].plot(lsigma, est_alpha[:, k], color=cm.jet(k / len(lsigma)))
            ax_both[1, 1].plot(lsigma, est_sigma[:, k], color=cm.jet(k / len(lsigma)))
        ax_both[0, 1].set_xlabel('sigma')
        ax_both[1, 1].set_xlabel('sigma')
        ax_both[0, 1].set_ylabel('alpha_est')
        ax_both[1, 1].set_ylabel('sigma_est')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--sigma0",action="store_true")
    parser.add_argument("--alpha",action="store_true")
    parser.add_argument("--both",action="store_true")
    parser.add_argument("--load",action="store_true")
    args = parser.parse_args()

    main()
