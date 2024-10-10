#!/usr/bin/env python3

import argparse
import numpy as np
import itk
import matplotlib.pyplot as plt

def main():
    print(args)

    source_itk = itk.imread(args.source)
    source_np = itk.array_from_image(source_itk)

    img1_itk = itk.imread(args.img1)
    img1_np = itk.array_from_image(img1_itk)

    img2_itk = itk.imread(args.img2)
    img2_np = itk.array_from_image(img2_itk)


    # id_img = [[63,73,71,86],[76,73,64,77], [76,73,49,63]]
    # id_src = [[113,146,181,238],[165,146,154,206], [165,146,94,149]]

    # id_img = [[76,73,64,77],[63,73,71,86], [76,73,49,63]]
    id_src = [[113,146,181,238],[165,146,154,206],[165,146,94,149],
              [115,146,77,111],[63,146,112,134],[63,146,171,187]]


    space = []
    last = 0
    for k in range(len(id_src)):
        space.append(np.arange(last, last+id_src[k][3] - id_src[k][2]))
        last += id_src[k][3] - id_src[k][2]+5

    fig,ax = plt.subplots()
    for k in range(len(id_src)):
        if k==0:
            ax.plot(space[k],source_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="black", label='source')
            ax.plot(space[k],img1_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="blue", label='rec noRM')
            ax.plot(space[k],img2_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="darkorange", label='rec RM')
        else:
            ax.plot(space[k],source_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="black")
            ax.plot(space[k],img1_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="blue")
            ax.plot(space[k],img2_np[id_src[k][0],id_src[k][1],id_src[k][2]:id_src[k][3]], color="darkorange")

    plt.legend(fontsize=15)
    plt.grid(False)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--img1")
    parser.add_argument("--img2")
    args = parser.parse_args()

    main()
