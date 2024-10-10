#!/usr/bin/env python3


import itk
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import argparse
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

# dictionnary whith spheres labels as keys and their corresponding radius as value.
dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
}


def main():
    print(args)

    legend = args.l
    color = args.c
    recons_img = args.images
    sources = args.sources
    labels=args.labels
    labels_json = args.labels_json
    fontsize = 30
    if len(legend)>0:
        assert(len(legend)==len(recons_img))
    else:
        legend = recons_img

    if len(color)>0:
        assert (len(color)==len(recons_img))
    else:
        color = ['green', 'blue', 'orange', 'red', 'black', 'magenta', 'cyan','yellow','brown','purple','pink','teal','gold','navy','olive','maroon','gray','lime','indigo','beige','turquoise']


    img_labels = itk.imread(labels)
    np_labels = itk.array_from_image(img_labels)

    json_labels_filename = labels_json
    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    # background mask
    if "background" in json_labels.keys():
        background_mask = (np_labels==json_labels['background'])
    elif "iec" in json_labels.keys():
        background_mask = (np_labels==json_labels['iec'])
    else:
        print(f"ERROR : background not found in --json_labels. Keys are : {json_labels.keys()}")
        exit(0)

    fig_RC,ax_RC = plt.subplots()
    ax_RC.set_xlabel('Sphere Volume (mL)', fontsize=fontsize)
    ax_RC.set_ylabel('Recovery Coefficient (RC)', fontsize=fontsize)

    fig_RMS,ax_RMS = plt.subplots()
    ax_RMS.set_xlabel('Sphere Volume (mL)', fontsize=fontsize)
    ax_RMS.set_ylabel('RMS', fontsize=fontsize)

    plt.rcParams["savefig.directory"] = os.getcwd()

    for img_num,(src_file, img_file) in enumerate(zip(sources,recons_img)):
        img_src = itk.imread(src_file)
        np_src = itk.array_from_image(img_src)
        if args.norm == "sum":
            np_src_normed = np_src / np_src.sum()
        elif args.norm == "sum_bg":
            np_src_normed = np_src / np_src[np_labels > 0].sum()
        else:
            np_src_normed = np_src

        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)
        if args.norm == "sum":
            np_recons_normalized = np_recons / np_recons.sum()
        elif args.norm=="sum_bg":
            np_recons_normalized = np_recons / np_recons[np_labels > 0].sum()
        elif args.norm=="CF":
            np_recons_normalized = np_recons*337
        else:
            print("no norm")
            np_recons_normalized = np_recons

        dict_sphereslabels_RC = {}
        dict_sphereslabels_RMS = {}
        print(img_file)


        for sph_label in dict_sphereslabels_radius:
            mean_act_src = (np_src_normed[np_labels==json_labels[sph_label]]).mean()
            mean_act_img = (np_recons_normalized[np_labels==json_labels[sph_label]]).mean()
            mean_act_img_bg = (np_recons_normalized[background_mask]).mean()
            std_act_img_bg = (np_recons_normalized[background_mask]).std()
            # dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src
            # dict_sphereslabels_RC[sph_label] = 1 - np.abs((np_recons_normalized-np_src_normed)[np_labels==json_labels[sph_label]]).mean()
            dict_sphereslabels_RC[sph_label] = 1 - (np.abs(np_recons_normalized-np_src_normed)[np_labels==int(json_labels[sph_label])]).mean()/(np.abs(np_src_normed[np_labels==int(json_labels[sph_label])])).mean()
            dict_sphereslabels_RMS[sph_label] = (mean_act_img - mean_act_img_bg) / std_act_img_bg

        # for lbl_name,lbl_id in json_labels.items():
        #     # mean_act_src = (np_src_normed[np_labels==json_labels[sph_label]]).mean()
        #     # mean_act_img = (np_recons_normalized[np_labels==json_labels[sph_label]]).mean()
        #     # mean_act_img_bg = (np_recons_normalized[background_mask]).mean()
        #     # std_act_img_bg = (np_recons_normalized[background_mask]).std()
        #     # dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src
        #     # dict_sphereslabels_RC[sph_label] = 1 - np.abs((np_recons_normalized-np_src_normed)[np_labels==json_labels[sph_label]]).mean()
        #     dict_sphereslabels_RC[sph_label] = (np.abs(np_recons_normalized-np_src)[np_labels==int(lbl_id)]).mean()
        #     dict_sphereslabels_RMS[sph_label] = 1


        x,y_RC,y_RMS =[],[],[]
        for sph_label in dict_sphereslabels_RC:
            x.append(4/3* np.pi * (dict_sphereslabels_radius[sph_label]/2)**3 * 1/1000)
            y_RC.append(dict_sphereslabels_RC[sph_label])
            y_RMS.append(dict_sphereslabels_RMS[sph_label])


        print(y_RC)

        ax_RC.plot(x,y_RC, '-o',markersize = 10, linewidth = 4, color = color[img_num], label = legend[img_num])
        ax_RMS.plot(x,y_RMS, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])


    ax_RC.plot(x, [1 for _ in x], '--', color = 'grey')
    ax_RC.legend(fontsize=15)
    ax_RC.tick_params(axis='x', labelsize=20)
    ax_RC.tick_params(axis='y', labelsize=20)
    ax_RMS.legend(fontsize=12)


    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("images", nargs='+')
    parser.add_argument("--labels")
    parser.add_argument("--labels-json")
    parser.add_argument("--sources", nargs='+', help = "SAME ORDER AS IMAGES")
    parser.add_argument("--norm")
    parser.add_argument("-l", nargs='+')
    parser.add_argument("-c", nargs='+', default = [])
    args = parser.parse_args()
    main()

