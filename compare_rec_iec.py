#!/usr/bin/env python3
import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt

dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }


spheres_loc = {1: {'l':41, 'c': [44,45]},
               2: {'l':41, 'c': [29,30,31]},
               3: {'l':28, 'c': [21,22, 23, 24, 25]},
               4: {'l': 16, 'c': np.arange(28,33)},
               5: {'l': 16, 'c': np.arange(41,48)},
               6: {'l':28, 'c': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]}}


def main():
    slicei=150
    print(args)

    

    src = itk.array_from_image(itk.imread(args.source))
    
    vmin,vmax = 0,src.max()*1.30

    # fig_img, ax_img = plt.subplots(2,3)
    # ax_img[0,0].imshow(src[:,slicei,:], vmin=vmin, vmax=vmax)
    fig_img, ax_img = plt.subplots(2,2)
    ax_img[0,0].axis('off')
    for (rec_img_fn, ax) in zip(args.images, ax_img[:,1:].reshape(-1)):
        rec_img = itk.array_from_image(itk.imread(rec_img_fn))
        im = ax.imshow(rec_img[:,slicei,:], vmin=vmin, vmax=vmax)
        ax.axis('off')
    ax_img[1,0].axis('off')
    fig_img.subplots_adjust(right=0.8)
    cbar_ax = fig_img.add_axes([0.85, 0.15, 0.05, 0.7])
    fig_img.colorbar(im, cax=cbar_ax)

    fig_prof,ax_prof = plt.subplots()

    space = 2
    abs = np.array([0])
    for sph, loc in spheres_loc.items():
        sph_abs = np.arange(abs[-1] + 1, abs[-1] + 1 + len(loc['c']) + 2 * space)
        abs = np.concatenate((abs, sph_abs))
        columns = np.concatenate((np.arange(loc['c'][0] - space, loc['c'][0]), loc['c'],
                                  np.arange(loc['c'][-1] + 1, loc['c'][-1] + 1 + space)))

        if sph==1:
            ax_prof.plot(sph_abs, src[loc['l'], slicei, columns], linestyle='dashed', label="src", color='black')
        else:
            ax_prof.plot(sph_abs, src[loc['l'], slicei, columns], linestyle='dashed', color='black')

    colors = ['green', 'blue', 'orange', 'red', 'pink']
    legends = args.legends.split(',')

    for i,rec_img in enumerate(args.images):
        rec_img = itk.array_from_image(itk.imread(rec_img))
        abs = np.array([0])
        for sph, loc in spheres_loc.items():
            sph_abs = np.arange(abs[-1] + 1, abs[-1] + 1 + len(loc['c']) + 2 * space)
            abs = np.concatenate((abs, sph_abs))
            columns = np.concatenate((np.arange(loc['c'][0] - space, loc['c'][0]), loc['c'],
                                      np.arange(loc['c'][-1] + 1, loc['c'][-1] + 1 + space)))

            if sph == 1:
                ax_prof.plot(sph_abs, rec_img[loc['l'], slicei, columns], label=legends[i], color = colors[i], linewidth = 2)
            else:
                ax_prof.plot(sph_abs, rec_img[loc['l'], slicei, columns], color = colors[i], linewidth = 2)



    plt.legend(fontsize = 20)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("images", nargs='+')
    parser.add_argument("--source")
    parser.add_argument("--legends")
    args = parser.parse_args()

    main()
