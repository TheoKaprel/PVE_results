#!/usr/bin/env python3

import click
import itk
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import FWHM_b
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
@click.option('--errors', is_flag = True, default = False)
@click.option('--auto')
@click.option('--ref', multiple=True)
@click.option('--mm', default='4')
def show_RC_curve(labels, labels_json, source, recons_img, legend,color, norm,errors, auto, ref, mm):
    if auto is not None:
        labels=os.path.join(auto, f'iec_labels_{mm}mm_c_rot.mhd')
        labels_json=os.path.join(auto, f'iec_labels_{mm}mm.json')
        source=os.path.join(auto,f'iec_src_bg_{mm}mm_c_rot.mhd') if (norm is not None)\
            else os.path.join(auto, f'iec_src_bg_{mm}mm_c_rot_scaled.mhd')
        recons_img=[os.path.join(auto, img_fn) for img_fn in ['iec_rec_noPVE_noPVC.mhd',
                                                              'iec_rec_PVE_PVC.mhd',
                                                              'iec_rec_PVE_noPVC.mhd']]
        legend=["noPVE-noPVC", "PVE-RM", "PVE-noPVC"]
        color = ["green", "blue", "red"]
        for one_ref in ref:
            recons_img.append(os.path.join(auto, f'iec_rec_DeepPVC_{one_ref}.mhd'))
            legend.append(f"PVE-DeepPVC")
            color.append("orange")


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

    np_src_norm = utils.calc_norm(np_src,norm)
    np_src = np_src / np_src_norm

    mean_bg_src = np.mean(np_src[background_mask])

    fig,ax_RC = plt.subplots()
    fig,ax_CNR = plt.subplots()
    ax_RC.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
    ax_RC.set_ylabel('RC', fontsize=18)
    ax_CNR.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
    ax_CNR.set_ylabel('CNR', fontsize=18)
    plt.rcParams["savefig.directory"] = os.getcwd()

    if errors:
        dict_err = {'NMAE': [], 'NRMSE':[], 'PSNR':[],'SSIM':[], 'labels':[]}

    for img_num,img_file in enumerate(recons_img):
        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)
        np_recons_norm = utils.calc_norm(np_recons, norm)
        np_recons_normalized = np_recons / np_recons_norm

        dict_sphereslabels_RC = {}
        dict_sphereslabels_CNR={}
        for sph_label in dict_sphereslabels_radius:
            mean_act_src = np.mean(np_src[np_labels == json_labels[sph_label]])
            mean_act_img = np.mean(np_recons_normalized[np_labels == json_labels[sph_label]])

            mean_bg_img = np.mean(np_recons_normalized[background_mask])

            dict_sphereslabels_RC[sph_label] = (mean_act_img - mean_bg_img) / (mean_act_src - mean_bg_src)
            dict_sphereslabels_CNR[sph_label] = utils.CNR(mask1=(np_labels == json_labels[sph_label]),
                                                          mask2=background_mask,
                                                          img=np_recons_normalized)

        x,y_RC,y_CNR = [],[],[]
        for sph_label in dict_sphereslabels_RC:
            x.append(dict_sphereslabels_radius[sph_label] / FWHM_b )
            y_RC.append(dict_sphereslabels_RC[sph_label])
            y_CNR.append(dict_sphereslabels_CNR[sph_label])

        ax_RC.plot(x,y_RC, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_CNR.plot(x,y_CNR, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])

        if errors:
            dict_err['NMAE'].append(utils.NMAE(img=np_recons_normalized,ref=np_src))
            dict_err['NRMSE'].append(utils.NRMSE(img=np_recons_normalized,ref=np_src))
            dict_err['PSNR'].append(utils.PSNR(img=np_recons_normalized,ref=np_src))
            dict_err['SSIM'].append(utils.SSIM(img=np_recons_normalized,ref=np_src))
            dict_err['labels'].append(legend[img_num])



    # ax_RC.set_title("Recovery Coefficients for the IEC phantom", fontsize = 18)
    ax_CNR.set_title("Contrast to Noise Ratio for IEC phantom", fontsize = 18)
    ax_RC.legend(fontsize=12)
    ax_CNR.legend()
    # plt.legend()

    if errors:
        colors=['grey','black', 'black', 'black']
        fig_e, ax_e = plt.subplots(2, 2, figsize=(20,20))
        ax_e[0,0].bar([k for k in range(len(dict_err['labels']))],dict_err['NMAE'], tick_label = dict_err['labels'], color = colors)
        ax_e[0,0].set_ylabel('NMAE', fontsize = 20, weight="bold")
        ax_e[0,1].bar([k for k in range(len(dict_err['labels']))],dict_err['NRMSE'], tick_label = dict_err['labels'], color = colors)
        ax_e[0,1].set_ylabel('NRMSE', fontsize = 20, weight="bold")
        ax_e[1,0].bar([k for k in range(len(dict_err['labels']))],dict_err['PSNR'], tick_label = dict_err['labels'], color = colors)
        ax_e[1,0].set_ylabel('PSNR', fontsize = 20, weight="bold")
        ax_e[1,1].bar([k for k in range(len(dict_err['labels']))],dict_err['SSIM'], tick_label = dict_err['labels'], color = colors)
        ax_e[1,1].set_ylabel('SSIM', fontsize = 20, weight="bold")
        font_prop={'weight' : 'bold', 'size' : 15, }
        for i in range(2):
            for j in range(2):
                ax_e[i,j].set_xticklabels(ax_e[i,j].get_xticklabels(), font_prop)
                ax_e[i,j].set_yticklabels(ax_e[i,j].get_yticks(), font_prop)
        ax_e[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_e[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_e[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax_e[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        for errtype in ["NRMSE", "NMAE", "PSNR", "SSIM"]:
            print(f'{errtype} : ')
            for i,lab in enumerate(dict_err['labels']):
                print(f"   {lab} : {round(dict_err[f'{errtype}'][i],4)}")
    plt.show()



if __name__ == '__main__':
    show_RC_curve()
