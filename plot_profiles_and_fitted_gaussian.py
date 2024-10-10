#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import itk
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def main():
    print(args)
    fig_x, ax_x = plt.subplots()
    spacing = 2.3976
    colors = ['green', 'orange', 'blue']
    legend = ["Experimental", "Opengate Simulation", "RTK Simulation"]
    for i,img in enumerate(args.inputs):
        array = itk.array_from_image(itk.imread(img))
        array = array/array.max()
        max_profile_x = array.sum(axis=0)
        x_data = np.linspace(-array.shape[0]*spacing/2, array.shape[0]*spacing/2, array.shape[0])
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
        x_fit = np.linspace(-array.shape[0]*spacing/2, array.shape[0]*spacing/2, 10*array.shape[0])
        y_fit = gaussian(x_fit, amplitude_fit, mean=mean_fit,stddev=stddev_fit)
        ax_x.scatter(x_data-mean_fit,max_profile_x, label=legend[i], color=colors[i])
        ax_x.plot(x_fit-mean_fit,y_fit, label=f'{legend[i]} (fitted)', color=colors[i])
    ax_x.set_xlim([-100,100])
    ax_x.set_xlabel("Distance (mm)", fontsize=18)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+")
    args = parser.parse_args()

    main()
