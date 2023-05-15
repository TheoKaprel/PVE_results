#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt

def main():
    print(args)

    ref_projs = itk.array_from_image(itk.imread(args.ref[0]))
    fig,ax = plt.subplots()

    abs = np.arange(0, ref_projs.shape[0])

    for proj_fn in args.projs_filenames:
        projs = itk.array_from_image(itk.imread(proj_fn))
        mse = np.mean((projs-ref_projs)**2,axis = (1,2))

        ax.plot(abs, mse, label = proj_fn)
        mse_sino = np.mean((projs-ref_projs)**2, axis=(0,1,2))
        print(f'Global MSE {proj_fn} : {mse_sino}')


    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, type = str, nargs=1)
    parser.add_argument("projs_filenames", type = str, nargs='*')
    parser.add_argument("--error", choices=['MAE', 'MSE'], default = 'MSE')
    args = parser.parse_args()

    main()
