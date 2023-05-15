#!/usr/bin/env python3

import argparse
import itk
import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def main():
    print(args)

    itk_projection = itk.imread(args.projection)
    spacing = itk_projection.GetSpacing()
    np_projection = itk.array_from_image(itk_projection)[0,:,:]

    # argmax = np.unravel_index(np.argmax(np_projection,axis=None),np_projection.shape)
    # maxval = np_projection[argmax]
    # p = np_projection * (np_projection>=maxval/2)
    # q0 = np.sum(p>0,axis=0)
    # q1 = np.sum(q0>0)
    # print(q1.shape)
    # print(q1)
    # fig,ax = plt.subplots()
    # ax.imshow(np_projection)
    # plt.show()
    fig,ax = plt.subplots()
    ax.imshow(np_projection)

    data = np_projection
    params = fitgaussian(data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)

    # ax = plt.gca()
    (height, x, y, width_x, width_y) = params

    print(params)

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" %(x, y, width_x, width_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("projection")
    args = parser.parse_args()

    main()
