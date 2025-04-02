#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt


def bresenham_3d(x1, x2):
    """
    Compute the voxel indices along a line from x1 to x2 in 3D space using Bresenham's algorithm.

    Parameters:
        x1: np.ndarray
            Starting point of the line (shape: (3,))
        x2: np.ndarray
            Ending point of the line (shape: (3,))

    Returns:
        List of voxel indices (each index is a tuple (x, y, z)).
    """
    x1, x2 = np.array(x1, dtype=int), np.array(x2, dtype=int)
    dx, dy, dz = np.abs(x2 - x1)
    sx = 1 if x2[0] > x1[0] else -1
    sy = 1 if x2[1] > x1[1] else -1
    sz = 1 if x2[2] > x1[2] else -1
    if dx >= dy and dx >= dz:  # x dominant
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        x, y, z = x1
        indices = [(x, y, z)]

        for _ in range(dx):
            x += sx
            if p1 >= 0:
                y += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            indices.append((x, y, z))
    elif dy >= dx and dy >= dz:  # y dominant
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        x, y, z = x1
        indices = [(x, y, z)]
        for _ in range(dy):
            y += sy
            if p1 >= 0:
                x += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            indices.append((x, y, z))

    else:  # z dominant
        p1 = 2 * dx - dz
        p2 = 2 * dy - dz
        x, y, z = x1
        indices = [(x, y, z)]
        for _ in range(dz):
            z += sz
            if p1 >= 0:
                x += sx
                p1 -= 2 * dz
            if p2 >= 0:
                y += sy
                p2 -= 2 * dz
            p1 += 2 * dx
            p2 += 2 * dy
            indices.append((x, y, z))

    return indices

def main():
    print(args)

    fig,ax = plt.subplots()

    for k,img_fn in enumerate(args.images):
        img = itk.imread(img_fn)
        array = itk.array_from_image(img)*337/1000
        indices = bresenham_3d(x1 = args.position1, x2 = args.position2)
        prof = [array[indices[k]] for k in range(len(indices))]
        x = np.arange(0,len(prof))*4.7952/10
        ax.plot(x, prof, label = args.l[k], color = args.c[k], linewidth = 4)


    ax.set_ylabel("MBq/mL", fontsize = 20)
    ax.set_xlabel("Position (cm)", fontsize = 20)
    ax.legend(fontsize = 20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.rcParams["savefig.directory"] = "/export/home/tkaprelian/Desktop/MANUSCRIPT/CHAP5_ALT_DL_PVC"

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--position1", nargs=3, type = int)
    parser.add_argument("--position2", nargs=3, type = int)
    parser.add_argument("-l", nargs='+')
    parser.add_argument("-c", nargs='+', default = [])
    args = parser.parse_args()

    main()
