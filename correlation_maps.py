#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import itk

def main():
    print(args)
    source = itk.array_from_image(itk.imread(args.source))
    fig,ax = plt.subplots(1,len(args.images))
    for img_i,img in enumerate(args.images):
        image = itk.array_from_image(itk.imread(img))
        ax[img_i].scatter(image.ravel(), source.ravel(), s = 2, color = 'blue')
        ax[img_i].plot(np.linspace(0,200), np.linspace(0,200), color = 'red')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+')
    parser.add_argument("--source")
    args = parser.parse_args()

    main()
