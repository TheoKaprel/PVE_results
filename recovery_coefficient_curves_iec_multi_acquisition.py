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

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import get_FWHM_b
import utils

def main():
    print(args)
    show_RC_curve(folders=args.folders)


def show_RC_curve(folders):
    # dictionnary whith spheres labels as keys and their corresponding radius as value.
    dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }

    b = 280
    FWHM_b = get_FWHM_b(machine="siemens-intevo", b=b)

    list_markers = [".", "v", "^"]

    fig_RC,ax_RC = plt.subplots()
    fig_MSE,ax_MSE = plt.subplots()

    for folder,marker in zip(folders,list_markers):
        labels_itk=itk.imread(os.path.join(folder, 'data/labels.mhd'))
        labels=itk.array_from_image(labels_itk)
        labels_json = open(os.path.join(folder, 'data/labels.json')).read()
        labels_json = json.loads(labels_json)

        src_itk = itk.imread(os.path.join(folder, 'data/src_kBq_mL.mhd'))
        src = itk.array_from_image(src_itk)
        src_n=src/src.sum()

        background_mask = (labels == labels_json['background'])

        mean_bg_src = np.mean(src_n[background_mask])

        list_rec_imgs=[os.path.join(folder, 'iter/rec_noPVC_5.mhd'),
                       os.path.join(folder, 'iter/rec_PVC_5.mhd'),
                       os.path.join(folder, 'iter/rec_PVCNet_407159_30_5.mhd')]

        list_colors=['red', 'blue', 'orange']

        for fn_rec_img,color in zip(list_rec_imgs,list_colors):
            rec_img_itk = itk.imread(fn_rec_img)
            rec_img = itk.array_from_image(rec_img_itk)
            rec_img_n = rec_img / rec_img.sum()

            for sph_label,sph_radius in dict_sphereslabels_radius.items():
                mean_act_src = np.mean(src_n[labels==labels_json[sph_label]])
                mean_act_rec = np.mean(rec_img_n[labels==labels_json[sph_label]])

                ax_RC.scatter(sph_radius, mean_act_rec / mean_act_src, color=color, marker=marker)
                sph_mask = labels==labels_json[sph_label]
                ax_MSE.scatter(sph_radius, np.mean((src_n[sph_mask] - rec_img_n[sph_mask])**2),  color=color, marker=marker, s = 20)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("folders", nargs='+')
    args = parser.parse_args()
    main()
