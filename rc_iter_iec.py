#!/usr/bin/env python3

import argparse
import os
import itk
import numpy as np
import matplotlib.pyplot as plt
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
    print(args)

    dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }

    fig_RC, ax_RC = plt.subplots()

    labels_itk = itk.imread(os.path.join(args.folder, 'data/labels.mhd'))
    labels = itk.array_from_image(labels_itk)
    labels_json = open(os.path.join(args.folder, 'data/labels.json')).read()
    labels_json = json.loads(labels_json)

    src_itk = itk.imread(os.path.join(args.folder, 'data/src_kBq_mL.mhd'))
    src = itk.array_from_image(src_itk)
    src_n = src / src.sum()

    list_imgs, list_iters = get_list_of_iter_img(args.basename)

    colors= ['lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'navy']


    for i,(sph_label, sph_radius) in enumerate(dict_sphereslabels_radius.items()):
        l_iter = []
        l_RC = []
        for fn_rec_img, iter in zip(list_imgs, list_iters):
            rec_img_itk = itk.imread(fn_rec_img)
            rec_img = itk.array_from_image(rec_img_itk)
            rec_img_n = rec_img / rec_img.sum()

            mean_act_src = np.mean(src_n[labels == labels_json[sph_label]])
            mean_act_rec = np.mean(rec_img_n[labels == labels_json[sph_label]])
            l_iter.append(iter)
            l_RC.append( mean_act_rec / mean_act_src)

        ax_RC.plot(l_iter, l_RC, color= colors[i], label=sph_label, marker='o')

    ax_RC.set_ylim([0,1.1])
    ax_RC.set_xlabel('iteration')
    ax_RC.set_ylabel('RC')
    ax_RC.set_title(args.title)
    plt.rcParams["savefig.directory"] = os.path.join(args.folder, 'fig')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--basename")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    main()
