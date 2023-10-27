#!/usr/bin/env python3

import argparse
import os.path

import itk
import matplotlib.pyplot as plt
import numpy as np

def get_list_of_iter_img(path_iterations):
    list_imgs,list_iters=[],[]
    for iter in range(1,100):
        file_iter=path_iterations.replace("%d", f'{iter}')
        if os.path.isfile(file_iter):
            list_imgs.append(file_iter)
            list_iters.append(iter)
    return list_imgs,list_iters


def main():
    print(args)
    srcitk = itk.imread(args.source)
    src = itk.array_from_image(srcitk)
    srcn = src/src.sum()


    fig,ax= plt.subplots(1,2)

    for k,basename in enumerate(args.imgs):
        l_mse = []
        l_mae = []

        list_imgs, list_iters = get_list_of_iter_img(basename)
        legend = os.path.basename(basename)

        for iter,img in zip(list_iters,list_imgs):
            imgitk = itk.imread(img)
            imgnp = itk.array_from_image(imgitk)
            imgn = imgnp/imgnp.sum()
            l_mae.append(np.mean(np.abs(imgn - srcn)))
            l_mse.append(np.sqrt(np.mean((imgn - srcn)**2)))

        ax[0].plot(l_mse, label = legend)
        ax[1].plot(l_mae, label = legend)

    ax[0].legend()
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--imgs", nargs='*')
    args = parser.parse_args()
    main()