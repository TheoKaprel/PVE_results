#!/usr/bin/env python3

import argparse
import itk
import matplotlib.pyplot as plt
import numpy as np


def main():
    print(args)
    srcitk = itk.imread(args.source)
    src = itk.array_from_image(srcitk)
    srcn = src/src.sum()

    legend = args.legend



    fig,ax= plt.subplots(1,2)

    for k,imgs in enumerate(args.imgs):
        l_mse = []
        l_mae = []
        for img in imgs:
            imgitk = itk.imread(img)
            imgnp = itk.array_from_image(imgitk)
            imgn = imgnp/imgnp.sum()
            l_mae.append(np.mean(np.abs(imgn - srcn)))
            l_mse.append(np.sqrt(np.mean((imgn - srcn)**2)))
        ax[0].plot(l_mse, label = legend[k])
        ax[1].plot(l_mae, label = legend[k])

    ax[0].legend()
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--imgs",action='append', nargs='*')
    parser.add_argument("--legend", nargs='*')
    args = parser.parse_args()
    main()