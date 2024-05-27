#!/usr/bin/env python3

import itk
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import argparse
import sys
import matplotlib.cm as cm

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import get_FWHM_b
import utils

def main():
    print(args)
    find_best_k(pw=args.pw,
                sw=args.sw,
                labels=args.labels,
                  labels_json=args.labels_json,
                  source = args.source,
                norm=args.norm)


def find_best_k(pw, sw, labels, labels_json, source, norm):
    dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }

    # open image labels
    img_labels = itk.imread(labels)
    np_labels = itk.array_from_image(img_labels)
    json_labels_filename = labels_json
    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    background_mask = (np_labels==json_labels['background'])
    img_src = itk.imread(source)
    np_src = itk.array_from_image(img_src)
    assert ((np_labels.shape == np_src.shape))

    np_src_norm = utils.calc_norm(np_src,norm)
    np_src_normed = np_src / np_src_norm

    mean_bg_src = np.mean(np_src[background_mask])

    array_pw = itk.array_from_image(itk.imread(pw))
    array_sw = itk.array_from_image(itk.imread(sw))

    list_k_values=np.linspace(0, 1.5, 50)

    fig,ax = plt.subplots()
    lmse,lmae=[],[]
    best_k=0
    y_best=[0]
    for i,k in enumerate(list_k_values):
        dict_sphereslabels_RC={}

        np_recons=array_pw - k * array_sw

        # np_recons_norm = utils.calc_norm(np.ma.masked_where(np_labels==0, np_recons,copy=True), norm)
        np_recons_norm = utils.calc_norm(np_recons, norm)
        np_recons_normalized = np_recons / np_recons_norm


        for sph_label in dict_sphereslabels_radius:
            mean_act_src = (np_src_normed[np_labels==json_labels[sph_label]]).mean()
            mean_act_img = (np_recons_normalized[np_labels==json_labels[sph_label]]).mean()
            dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src

        x, y_RC = [], []
        for sph_label in dict_sphereslabels_RC:
            x.append(4 / 3 * np.pi * (dict_sphereslabels_radius[sph_label] / 2) ** 3 * 1 / 1000)
            y_RC.append(dict_sphereslabels_RC[sph_label])
        ax.plot(x, y_RC, label=f'k= {k}', color=cm.jet(i/len(list_k_values)))

        if abs(y_RC[-1]-1)<abs(y_best[-1]-1):
            y_best=y_RC
            best_k = k

    print(f"Best k : {best_k}")

    ax.legend()
    ax.plot(x, [1 for _ in x], color='grey', linestyle=':', label='one')
    ax.plot(x, y_best, color='green', linestyle='-', label='BEST', linewidth=2)


    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pw")
    parser.add_argument("--sw")
    parser.add_argument("--source")
    parser.add_argument("--labels")
    parser.add_argument("--labels-json")
    parser.add_argument("--norm")
    args = parser.parse_args()

    main()
