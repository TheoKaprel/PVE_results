#!/usr/bin/env python3

import argparse
import os.path

import itk
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import json


def get_list_of_iter_img(path_iterations):
    list_imgs,list_iters=[],[]
    for iter in range(1,100):
        file_iter=path_iterations.replace("%d", f'{iter}')
        if os.path.isfile(file_iter):
            list_imgs.append(file_iter)
            list_iters.append(iter)
    return list_imgs,list_iters


def main():
    dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }


    print(args)

    # open image labels
    img_labels = itk.imread(args.labels)
    np_labels = itk.array_from_image(img_labels)
    # open json labels
    json_labels_filename = args.labels_json

    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    # background mask
    background_mask = (np_labels==json_labels['background'])


    srcitk = itk.imread(args.source)
    src = itk.array_from_image(srcitk)
    srcn = src/src.sum()


    fig,ax= plt.subplots(1,3)

    for k,basename in enumerate(args.imgs):
        l_mse = []
        l_mae = []
        l_mRC,l_SNR=[],[]

        list_imgs, list_iters = get_list_of_iter_img(basename)
        legend = os.path.basename(basename)

        fig_rc, ax_rc = plt.subplots()
        for iter,img in zip(list_iters,list_imgs):
            imgitk = itk.imread(img)
            imgnp = itk.array_from_image(imgitk)
            imgn = imgnp/imgnp.sum()
            l_mae.append(np.mean(np.abs(imgn - srcn)))
            l_mse.append(np.sqrt(np.mean((imgn - srcn)**2)))

            l_RC = []
            x_vol = []
            for sph_label in dict_sphereslabels_radius:
                mean_act_src = (srcn[np_labels == json_labels[sph_label]]).mean()
                mean_act_img = (imgn[np_labels == json_labels[sph_label]]).mean()
                l_RC.append(mean_act_img / mean_act_src)

                x_vol.append((4 / 3 * np.pi * (dict_sphereslabels_radius[sph_label] / 2) ** 3 * 1 / 1000))


            ax_rc.plot(x_vol, l_RC, color = cm.Blues(iter/len(list_iters)), label=f'{iter}')

            l_mRC.append(sum(l_RC)/len(l_RC))
            bg_img = imgn[background_mask]
            l_SNR.append(np.mean(bg_img)/np.std(bg_img))

        ax_rc.set_title(f'RC {ax_rc}')
        ax_rc.legend()
        if args.save:
            fn = basename.replace('.mhd', '_mse.npy')
            np.save(fn,np.array(l_mse))

            fn = basename.replace('.mhd', '_mae.npy')
            np.save(fn, np.array(l_mae))

            fn = basename.replace('.mhd', '_snr.npy')
            np.save(fn, np.array(l_SNR))

            fn = basename.replace('.mhd', '_mrc.npy')
            np.save(fn, np.array(l_mRC))

        ax[0].plot(l_mse, label = legend)
        ax[1].plot(l_mae, label = legend)
        ax[2].scatter(l_mRC,l_SNR,marker='o',s=[n for n in range(len(l_mRC))])
        ax[2].plot(l_mRC,l_SNR, label=legend)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel("RC")
    ax[2].set_ylabel("SNR")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--imgs", nargs='*')
    parser.add_argument("--labels")
    parser.add_argument("--labels-json")
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    main()