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
    if "mlem" not in path_iterations:
        list_iters = [8*i for i in list_iters]
    return list_imgs,list_iters


def main():
    print(args)

    # open image labels
    img_labels = itk.imread(args.labels)
    np_labels = itk.array_from_image(img_labels)
    # open json labels
    json_labels_filename = args.labels_json
    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    # background mask
    body_mask = (np_labels>0)
    background_mask = (np_labels==1)

    srcitk = itk.imread(args.source)
    src = itk.array_from_image(srcitk)
    # srcn = src/src.sum()
    srcn = src


    fig,ax= plt.subplots(1,3)

    for k,basename in enumerate(args.imgs):
        l_mse = []
        l_mae = []
        l_mRC,l_SNR=[],[]

        list_imgs, list_iters = get_list_of_iter_img(basename)
        legend = os.path.basename(basename)

        for iter,img in zip(list_iters,list_imgs):
            imgitk = itk.imread(img)
            imgnp = itk.array_from_image(imgitk)
            # imgn = imgnp/imgnp[background_mask].sum()
            # imgn = imgnp/imgnp[background_mask].sum()*srcn.sum()
            # imgn = imgnp
            imgn = imgnp*337
            l_mae.append(np.mean(np.abs(imgn[body_mask] - srcn[body_mask]))/np.mean(np.abs(src[body_mask])))
            l_mse.append(np.sqrt(np.mean((imgn[body_mask] - srcn[body_mask])**2))/np.mean(np.abs(src[body_mask])))

            l_RC = []
            l_AC_lesion = []
            for organ,lbl in json_labels.items():
                mean_act_src = (srcn[np_labels == int(lbl)]).mean()
                mean_act_img = (imgn[np_labels == int(lbl)]).mean()
                l_RC.append(mean_act_img / mean_act_src)
                l_AC_lesion.append(mean_act_img)
            AR = sum([1 - abs(rc -1) for rc in l_RC])/len(l_RC)
            l_mRC.append(AR)
            bg_img = imgn[background_mask]
            # l_SNR.append(np.mean(bg_img)/np.std(bg_img))
            C_lesions = sum(l_AC_lesion) / len(l_AC_lesion)
            l_SNR.append((C_lesions-np.mean(bg_img))/np.std(bg_img))

        # if args.save:
        #     fn = basename.replace('.mhd', '_mse.npy')
        #     np.save(fn,np.array(l_mse))
        #
        #     fn = basename.replace('.mhd', '_mae.npy')
        #     np.save(fn, np.array(l_mae))
        #
        #     fn = basename.replace('.mhd', '_snr.npy')
        #     np.save(fn, np.array(l_SNR))
        #
        #     fn = basename.replace('.mhd', '_mrc.npy')
        #     np.save(fn, np.array(l_mRC))

        ax[0].plot(list_iters,l_mse, label = legend)
        ax[1].plot(list_iters,l_mae, label = legend)
        ax[0].set_title("MSE")
        ax[1].set_title("MAE")
        ax[2].scatter(l_mRC,l_SNR,marker='o',s=20)
        # ax[2].scatter(l_mRC,l_SNR,marker='o',s=10)
        ax[2].plot(l_mRC,l_SNR, label=legend)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel("RC")
    ax[2].set_ylabel("CNR")
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