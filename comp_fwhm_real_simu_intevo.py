#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from itk import RTK as rtk
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
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

def calc_fwhm_gaussian_fit(array, spacing):

    # fwhm_x,fwhm_y = calc_fwhm(array, spacing)

    max_profile_x = array.sum(axis=0)
    max_profile_y = array.sum(axis=1)

    x_data = np.linspace(-array.shape[0]*spacing/2, array.shape[0]*spacing/2, array.shape[0])
    # y_data = 3 * np.exp(-(x_data - 2) ** 2 / (2 * 1.5 ** 2)) + np.random.normal(0, 0.2, x_data.size)

    # Define the Gaussian function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    # Fit the Gaussian function to the data
    initial_amplitude = max_profile_x.max()

    half_max = initial_amplitude / 2
    indices_above_half_max = np.where(max_profile_x > half_max)[0]
    if len(indices_above_half_max) >= 2:
        initial_stddev = (x_data[indices_above_half_max[-1]] - x_data[indices_above_half_max[0]]) / (
                    2 * np.sqrt(2 * np.log(2)))
    else:
        initial_stddev = 1.0  # Fallback if we can't estimate the stddev

    initial_guess = [initial_amplitude, x_data[np.argmax(max_profile_x)], initial_stddev]  # Initial guess for the parameters
    params, covariance = curve_fit(gaussian, x_data, max_profile_x, p0=initial_guess)

    # Extract the fitted parameters
    amplitude_fit, mean_fit, stddev_fit = params
    fwhm_x = stddev_fit * 2 * np.sqrt(2*np.log(2))



    # Fit the Gaussian function to the data
    initial_amplitude = max_profile_y.max()
    half_max = initial_amplitude / 2
    indices_above_half_max = np.where(max_profile_y > half_max)[0]
    if len(indices_above_half_max) >= 2:
        initial_stddev = (x_data[indices_above_half_max[-1]] - x_data[indices_above_half_max[0]]) / (
                    2 * np.sqrt(2 * np.log(2)))
    else:
        initial_stddev = 1.0  # Fallback if we can't estimate the stddev
    initial_guess = [initial_amplitude, x_data[np.argmax(max_profile_y)], initial_stddev]  # Initial guess for the parameters
    params, covariance = curve_fit(gaussian, x_data, max_profile_y, p0=initial_guess)
    # Extract the fitted parameters
    amplitude_fit, mean_fit, stddev_fit = params
    fwhm_y = stddev_fit * 2 * np.sqrt(2*np.log(2))

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

            # fwhm_x_1,fwhm_y_1 = calc_fwhm(array=np_array[0,:,:],spacing=spacing)
            # fwhm_x_2,fwhm_y_2 = calc_fwhm(array=np_array[1,:,:],spacing=spacing)

            fwhm_x_1,fwhm_y_1 = calc_fwhm_gaussian_fit(array=np_array[0,:,:],spacing=spacing)
            fwhm_x_2,fwhm_y_2 = calc_fwhm_gaussian_fit(array=np_array[1,:,:],spacing=spacing)

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
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/exp_100mm.mhd")
            elif "15CM" in dicom:
                ldistance_1.append(150)
                ldistance_2.append(150)
            elif "25CM" in dicom:
                ldistance_1.append(250)
                ldistance_2.append(250)
            elif "30CM" in dicom:
                ldistance_1.append(300)
                ldistance_2.append(300)
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/exp_300mm.mhd")


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


        ax.scatter(ldistance_1,lfwhm_x_1, marker="o", label='Experimental (horizontal)', color='green', s = 30)
        ax.scatter(ldistance_1,lfwhm_y_1, marker="s",label='Experimental (vertical)', color='green', s = 30)
        ax.scatter(ldistance_2,lfwhm_x_2, marker="o", color='green', s = 30)
        ax.scatter(ldistance_2,lfwhm_y_2, marker="s", color='green', s = 30)

        # Create a Linear Regression object
        regressor = LinearRegression()
        # Tran the model using the training sets
        X_fit = np.concatenate((ldistance_1,ldistance_1,ldistance_2,ldistance_2)).reshape(-1,1)
        Y_fit = np.concatenate((lfwhm_x_1,lfwhm_y_1,lfwhm_x_2,lfwhm_y_2)).reshape(-1,1)
        regressor.fit(X_fit, Y_fit)
        # Make predictions using the testing set
        X_pred = np.linspace(30,300).reshape(-1,1)
        Y_pred = regressor.predict(X_pred)
        # The coefficients
        print('Coefficients EXP : ', regressor.coef_, regressor.intercept_)
        # Plot outputs
        plt.plot(X_pred, Y_pred,linestyle="dashed",color='green',label="Measured (fitted)", linewidth=2)

    dists = [50, 100, 150, 200, 250, 300, 350]
    if args.simu1 is not None:
        lfwhm_x,lfwhm_y=[],[]
        for fnames in args.simu1:
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            # fwhm_x, fwhm_y= calc_fwhm_gaussian_fit(array=np_array[0, :, :], spacing=2.3976)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[0, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)

            if "100mm" in fnames:
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/rtk_100mm.mhd")
            elif "300mm" in fnames:
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/rtk_300mm.mhd")


        ax.scatter(dists,lfwhm_x, marker='o', label='RTK Simulation (horizontal)', color = 'green')
        ax.scatter(dists,lfwhm_y, marker='s', label='RTK Simulation (vertical)', color = 'green')



    dists = [50,100, 150, 200, 250, 300,350]
    # dists = [d-24 for d in dists]
    if args.simu2 is not None:
        lfwhm_x,lfwhm_y=[],[]
        for fnames in args.simu2:
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)


            # fwhm_x, fwhm_y= calc_fwhm_gaussian_fit(array=np_array[4, :, :], spacing=2.3976)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[1, :, :], spacing=2.3976)

            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)

            if "100mm" in fnames:
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/opengate_100mm.mhd")
            elif "300mm" in fnames:
                itk.imwrite(itk.image_from_array(np_array[0, :, :]/np_array[0, :, :].max()), "/export/home/tkaprelian/Desktop/PVE/datasets/PSF_determination/tc/figure/opengate_300mm.mhd")


        # Create a Linear Regression object
        regressor = LinearRegression()
        # Tran the model using the training sets
        dists = np.array(dists)
        lfwhm_x = np.array(lfwhm_x)
        lfwhm_y = np.array(lfwhm_y)
        X_fit = np.concatenate((dists,dists)).reshape(-1, 1)
        Y_fit = np.concatenate((lfwhm_x, lfwhm_y)).reshape(-1, 1)
        regressor.fit(X_fit, Y_fit)
        # Make predictions using the testing set
        X_pred = np.linspace(30, 350).reshape(-1, 1)
        Y_pred = regressor.predict(X_pred)
        print('Coefficients SIMU : ', regressor.coef_, regressor.intercept_)


        X_pred = np.linspace(30,350)
        # Y_analytic_colli = 3.704 + 0.071 * X_pred
        Y_analytic_colli = 2.94 + 0.07585648356116731 *X_pred
        Rint = 3.9*np.sqrt(140.5)/np.sqrt(208)
        Y_analytic = np.sqrt(Rint**2 + Y_analytic_colli**2)
        ax.plot(X_pred, Y_analytic,linestyle="dashed",color='red',label="Analytical", linewidth=2)


        ax.scatter(dists,lfwhm_x, marker='o', label='Opengate Simulation (vertical)', color = 'orange')
        ax.scatter(dists,lfwhm_y, marker='s', label='Opengate Simulation (horizontal)', color = 'orange')
        ax.plot(X_pred, Y_pred,linestyle="dashed",color='orange',label="Opengate Simulation (fitted)", linewidth=2)


    dists = [50, 100, 150, 200, 250, 300, 350]
    if args.simu3 is not None:
        lfwhm_x,lfwhm_y=[],[]
        for fnames in args.simu3:
            img = itk.imread(fnames)
            np_array = itk.array_from_image(img)
            fwhm_x, fwhm_y= calc_fwhm(array=np_array[0, :, :], spacing=2.3976)
            lfwhm_x.append(fwhm_x)
            lfwhm_y.append(fwhm_y)


        ax.scatter(dists,lfwhm_x, marker='o', label='RTK Simulation (horizontal)', color = 'blue')
        ax.scatter(dists,lfwhm_y, marker='s', label='RTK Simulation (vertical)', color = 'blue')


    # if args.simu2 is not None:
    #     lfwhm_x,lfwhm_y=[],[]
    #     for fnames in args.simu2:
    #         img = itk.imread(fnames)
    #         np_array = itk.array_from_image(img)
    #         fwhm_x, fwhm_y= calc_fwhm(array=np_array[4, :, :], spacing=2.3976)
    #         lfwhm_x.append(fwhm_x)
    #         lfwhm_y.append(fwhm_y)
    #     ax.plot(dists,lfwhm_x, marker='o', label='simu Lu177-MEGP', color = 'orange')
    #     ax.plot(dists,lfwhm_y, marker='s', color = 'orange')

    # if args.simu3 is not None:
    #     lfwhm_x,lfwhm_y=[],[]
    #     dists=[]
    #     for fnames in args.simu3:
    #         i = fnames.find('mm')
    #         dists.append(float(fnames[i-3:i]))
    #         img = itk.imread(fnames)
    #         np_array = itk.array_from_image(img)
    #         fwhm_x, fwhm_y= calc_fwhm(array=np_array[0, :, :], spacing=2.3976)
    #         lfwhm_x.append(fwhm_x)
    #         lfwhm_y.append(fwhm_y)
    #     ax.plot(dists,lfwhm_x, marker='o', label='simu3 fwhm_x', color = 'blueviolet')
    #     ax.plot(dists,lfwhm_y, marker='s', label='simu3 fwhm_y', color = 'blueviolet')

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

    # ax.set_ylim([2, 20])
    ax.set_xlabel("Source to detector distance (mm)",fontsize=18)
    ax.set_ylabel("FWHM (mm)", fontsize=18)
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
