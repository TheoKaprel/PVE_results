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

    exp_alpha = 0.018254840198984965
    exp_sigma0 = 1.5812410668449028
    c =  2 * np.sqrt(2 * np.log(2))
    exp_alpha_fwhm, exp_sigma0_fwhm = exp_alpha * c , exp_sigma0 * c

    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]
    source = itk.imread(args.source, pixelType)
    dists = [100, 150, 200, 250, 300, 350]

    proj_size = [256, 256, len(dists)]
    proj_spacing = [2.3976, 2.3976, 1]
    proj_offset = [-(s - 1) * sp / 2 for s, sp in zip(proj_size, proj_spacing)]

    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for d in dists:
        geometry.AddProjection(d, 0, 0, proj_offset[0], proj_offset[1])

    output_proj = rtk.ConstantImageSource[imageType].New()
    output_proj.SetSpacing(proj_spacing)
    output_proj.SetOrigin(proj_offset)
    output_proj.SetSize(proj_size)
    output_proj.SetConstant(0.)

    forward_projector = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector.SetInput(0, output_proj.GetOutput())
    forward_projector.SetInput(1, source)
    forward_projector.SetGeometry(geometry)

    forward_projector.SetSigmaZero(args.sigma0)
    forward_projector.SetAlpha(args.alpha)
    forward_projector.Update()
    output_proj = forward_projector.GetOutput()
    output_proj_array = itk.array_from_image(output_proj)

    lfwhm = []
    for p in range(output_proj_array.shape[0]):
        fwhm_x,fwhm_y = calc_fwhm(array=output_proj_array[p, :, :], spacing=proj_spacing[0])
        fwhm = (fwhm_x+fwhm_y)/2
        lfwhm.append(fwhm)



    ldistance_1, ldistance_2  = [], []
    lfwhm_x_1, lfwhm_x_2,lfwhm_y_1, lfwhm_y_2=[], [], [], []
    lfwhm_1,lfwhm_2 = [],[]

    for dicom in args.dicomfiles:
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



    fig,ax = plt.subplots()
    ax.plot(dists, lfwhm, label='rtk')
    ax.plot(dists, [exp_sigma0_fwhm + exp_alpha_fwhm*d for d in dists], label='exp')
    ax.plot(ldistance_1,lfwhm_x_1, marker="o", label='fwhm_x_1')
    ax.plot(ldistance_1,lfwhm_y_1, marker="o", label='fwhm_y_1')
    ax.plot(ldistance_2,lfwhm_x_2, marker="o", label='fwhm_x_2')
    ax.plot(ldistance_2,lfwhm_y_2, marker="o", label='fwhm_y_2')


    if args.gagarfiles is not None:
        lfwhm_x_gagarf,lfwhm_y_gagarf=[],[]
        # gagarf_dists = [50,100,150,200,250,300]
        gagarf_dists = [380,400]
        for gagarfn in args.gagarfiles:
            img = itk.imread(gagarfn)
            np_array = itk.array_from_image(img)
            fwhm_x_gagarf, fwhm_y_gagarf = calc_fwhm(array=np_array[2, :, :], spacing=2.3976)
            lfwhm_x_gagarf.append(fwhm_x_gagarf)
            lfwhm_y_gagarf.append(fwhm_y_gagarf)
        ax.plot(gagarf_dists,lfwhm_x_gagarf, marker='o', label='fwhm_x_gagarf')
        ax.plot(gagarf_dists,lfwhm_y_gagarf, marker='o', label='fwhm_y_gagarf')


    ax.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--dicomfiles", nargs='*')
    parser.add_argument("--gagarfiles", nargs='*')
    parser.add_argument("--sigma0", type=float)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()

    main()
