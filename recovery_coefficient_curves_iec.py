#!/usr/bin/env python3

import click
import itk
import matplotlib.pyplot as plt
import numpy as np
import json

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--labels', help = 'ex: iec_labels.mhd. ie image containing the label of the object in each voxel. It is assumed that iec_labels.json is in the same folder as iec_labels.mhd')
@click.option('--source')
@click.option('--i', 'recons_img', multiple = True)
@click.option('-l','--legend', multiple = True)
@click.option('-c','--color', multiple = True)
@click.option('--norm')
def show_RC_curve(labels, source, recons_img, legend,color, norm):

    if len(legend)>0:
        assert(len(legend)==len(recons_img))
    else:
        legend = recons_img

    if len(color)>0:
        assert (len(color)==len(recons_img))
    else:
        color = ['green', 'blue', 'orange', 'red', 'black', 'magenta']

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
    if '.mhd' in labels:
        json_labels_filename = labels.replace('.mhd', '.json')
    elif '.mha' in labels:
        json_labels_filename = labels.replace('.mha', '.json')
    else:
        print('ERROR. Unable to load labels')
        exit(0)
    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)

    # open source image
    img_src = itk.imread(source)
    np_src = itk.array_from_image(img_src)

    assert ((np_labels.shape==np_src.shape))

    fig,ax = plt.subplots()
    ax.set_xlabel('Object radius (mm)')
    ax.set_ylabel('Recovery Coefficient')

    for img_num,img_file in enumerate(recons_img):
        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)


        dict_sphereslabels_RC = {}
        for sph_label in dict_sphereslabels_radius:
            mean_act_src = np.mean(np_src[np_labels == json_labels[sph_label]])
            mean_act_img = np.mean(np_recons[np_labels == json_labels[sph_label]])
            dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src

        x = []
        y = []
        for sph_label in dict_sphereslabels_RC:
            x.append(dict_sphereslabels_radius[sph_label])
            y.append(dict_sphereslabels_RC[sph_label])

        ax.plot(x,y, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])

    plt.legend()
    plt.show()







if __name__ == '__main__':
    show_RC_curve()