#!/usr/bin/env python3

import click
import itk
import matplotlib.pyplot as plt
import numpy as np
import json
import os


import utils

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--labels', help = 'ex: iec_labels.mhd. ie image containing the label of the object in each voxel. if labels-json is not provided, it is assumed that iec_labels.json is in the same folder as iec_labels.mhd')
@click.option('--labels-json')
@click.option('--source')
@click.option('--i', 'recons_img', multiple = True)
@click.option('-l','--legend', multiple = True)
@click.option('-c','--color', multiple = True)
@click.option('--norm')
@click.option('--title')
def show_RC_curve(labels, labels_json, source, recons_img, legend,color, norm, title):

    if len(legend)>0:
        assert(len(legend)==len(recons_img))
    else:
        legend = recons_img

    if len(color)>0:
        assert (len(color)==len(recons_img))
    else:
        color = ['green', 'blue','cyan', 'orange', 'red', 'black', 'magenta']

    # dictionnary whith spheres labels as keys and their corresponding radius as value.
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
    # open json labels
    if labels_json is None:
        if '.mhd' in labels:
            json_labels_filename = labels.replace('.mhd', '.json')
        elif '.mha' in labels:
            json_labels_filename = labels.replace('.mha', '.json')
        else:
            print('ERROR. Unable to load labels')
            exit(0)
    else:
        json_labels_filename = labels_json

    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    # background mask
    background_mask = (np_labels>1)
    for sph in dict_sphereslabels_radius:
        background_mask = background_mask * (np_labels!=json_labels[sph])

    # open source image
    img_src = itk.imread(source)
    np_src = itk.array_from_image(img_src)
    assert ((np_labels.shape == np_src.shape))

    if norm=='sum':
        np_src_norm = utils.calc_norm(np_src,norm)
        np_src = np_src / np_src_norm

    mean_bg_src = np.mean(np_src[background_mask])

    fig,ax = plt.subplots()
    ax.set_xlabel('Object radius (mm)', fontsize=18)
    ax.set_ylabel('contrast Recovery Coefficient', fontsize=18)
    plt.rcParams["savefig.directory"] = os.getcwd()

    for img_num,img_file in enumerate(recons_img):
        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)
        np_recons_norm = utils.calc_norm(np_recons, norm)
        np_recons_normalized = np_recons / np_recons_norm

        dict_sphereslabels_RC = {}
        for sph_label in dict_sphereslabels_radius:
            mean_act_src = np.mean(np_src[np_labels == json_labels[sph_label]])
            mean_act_img = np.mean(np_recons_normalized[np_labels == json_labels[sph_label]])

            mean_bg_img = np.mean(np_recons_normalized[background_mask])

            dict_sphereslabels_RC[sph_label] = (mean_act_img - mean_bg_img) / (mean_act_src - mean_bg_src)

        x = []
        y = []
        for sph_label in dict_sphereslabels_RC:
            x.append(dict_sphereslabels_radius[sph_label])
            y.append(dict_sphereslabels_RC[sph_label])

        ax.plot(x,y, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])

    if title:
        ax.set_title(title, fontsize = 18)

    plt.legend()
    plt.show()







if __name__ == '__main__':
    show_RC_curve()
