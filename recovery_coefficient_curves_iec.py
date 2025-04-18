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
    show_RC_curve(labels=args.labels,
                  labels_json=args.labels_json,
                  source = args.source,
                  recons_img=args.images,
                  legend=args.l,
                  color = args.c,
                  norm=args.norm,
                  errors=args.errors,
                  auto = args.auto,
                  ref = args.ref,
                  mm = args.mm)


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
        colors_bis = ["orange", "gold", "darkorange"]
        for j,one_ref in enumerate(ref):
            recons_img.append(os.path.join(auto, f'iec_rec_DeepPVC_{one_ref}.mhd'))
            legend.append(f"PVE-DeepPVC-{one_ref}")
            color.append(colors_bis[j])


    if len(legend)>0:
        assert(len(legend)==len(recons_img))
    else:
        legend = recons_img

    if len(color)>0:
        assert (len(color)==len(recons_img))
    else:
        color = ['green', 'blue', 'orange', 'red', 'black', 'magenta', 'cyan','yellow','brown','purple','pink','teal','gold','navy','olive','maroon','gray','lime','indigo','beige','turquoise']

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
    if 'background' in json_labels.keys():
        background_mask = (np_labels==json_labels['background'])
    else:
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
    print(f'mean bg : {mean_bg_src}')

    fig,ax_RC = plt.subplots()
    fig,ax_CRC = plt.subplots()
    fig,ax_CNR = plt.subplots()
    fig,ax_RMSD = plt.subplots()
    fig,ax_RMSE = plt.subplots()
    fig,ax_mAC = plt.subplots()
    ax_RC.set_xlabel('Sphere Diameter / FWHM (280mm)', fontsize=18)
    ax_CRC.set_xlabel('Sphere Diameter / FWHM (280mm)', fontsize=18)
    ax_RC.set_ylabel('RC', fontsize=18)
    ax_CRC.set_ylabel('CRC', fontsize=18)
    ax_CNR.set_xlabel('Sphere Diameter / FWHM (280mm)', fontsize=18)
    ax_CNR.set_ylabel('CNR', fontsize=18)
    ax_RMSD.set_xlabel('Sphere Diameter / FWHM (280mm)', fontsize=18)
    ax_RMSE.set_xlabel('Sphere Diameter / FWHM (280mm)', fontsize=18)
    ax_RMSD.set_ylabel('RMSD', fontsize=18)
    ax_RMSE.set_ylabel('local NMAE', fontsize=18)
    ax_mAC.set_ylabel('mean Activity', fontsize=18)
    plt.rcParams["savefig.directory"] = os.getcwd()

    if errors:
        dict_err = {'NMAE': [], 'NRMSE':[], 'PSNR':[],'SSIM':[], 'labels':[]}

    for img_num,img_file in enumerate(recons_img):
        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)

        np_recons_norm = utils.calc_norm(np.ma.masked_where(np_labels==0, np_recons,copy=True), norm)
        np_recons_normalized = np_recons / np_recons_norm

        dict_sphereslabels_RC = {}
        dict_sphereslabels_CRC = {}
        dict_sphereslabels_CNR={}
        dict_sphereslabels_RMSD={}
        dict_sphereslabels_RMSE = {}
        dict_sphereslabels_mAC = {}
        print(img_file)
        for sph_label in dict_sphereslabels_radius:
            mask_src = np.ma.masked_where(np_labels != json_labels[sph_label], np_src,copy=True)
            mask_img = np.ma.masked_where(np_labels != json_labels[sph_label], np_recons_normalized,copy=True)
            mean_act_src = mask_src.mean()
            mean_act_img = mask_img.mean()
            print(sph_label)
            print(f'mean act img : {mean_act_img}')
            print(f'mean act src : {mean_act_src}')
            print(f'RC: {mean_act_img / mean_act_src}')

            mean_bg_img = np.mean(np_recons_normalized[background_mask])

            dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src
            dict_sphereslabels_CRC[sph_label] = ((mean_act_img - mean_bg_img)/mean_bg_img) / ((mean_act_src - mean_bg_src)/mean_bg_src)
            dict_sphereslabels_CNR[sph_label] = utils.CNR(mask1=(np_labels == json_labels[sph_label]),
                                                          mask2=background_mask,
                                                          img=np_recons_normalized)
            # dict_sphereslabels_RMSD[sph_label] = utils.RMS(mask=(np_labels==json_labels[sph_label]), img=np_recons_normalized,src = np_src)
            dict_sphereslabels_RMSD[sph_label] = utils.RMS(mask=(np_labels==json_labels[sph_label]), img=np_recons_normalized)

            dict_sphereslabels_RMSE[sph_label] = utils.local_NMAE(mask=(np_labels==json_labels[sph_label]),img=np_recons_normalized, src = np_src)

            dict_sphereslabels_mAC[sph_label] = np.mean(np_recons[np_labels==json_labels[sph_label]]) / 2400 / (2.3976**3) / np.mean((np_src*np_src_norm)[np_labels==json_labels[sph_label]])

        global_CNR = utils.CNR(mask1=(np_labels==6) + (np_labels==8 ) + (np_labels==12) + (np_labels==13) + (np_labels==14) + (np_labels==15),
                               mask2 = background_mask, img = np_recons_normalized)
        print(f'global CNR : {global_CNR}')

        x,y_RC,y_CRC,y_CNR,y_RMSD,y_RMSE,y_mAC =[],[],[],[],[],[],[]
        for sph_label in dict_sphereslabels_RC:
            x.append(dict_sphereslabels_radius[sph_label] / FWHM_b )
            y_RC.append(dict_sphereslabels_RC[sph_label])
            y_CRC.append(dict_sphereslabels_CRC[sph_label])
            y_CNR.append(dict_sphereslabels_CNR[sph_label])
            y_RMSD.append(dict_sphereslabels_RMSD[sph_label])
            y_RMSE.append(dict_sphereslabels_RMSE[sph_label])
            y_mAC.append(dict_sphereslabels_mAC[sph_label])

        print(y_RC)
        ax_RC.plot(x,y_RC, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_CRC.plot(x,y_CRC, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_CNR.plot(x,y_CNR, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_RMSD.plot(x,y_RMSD, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_RMSE.plot(x,y_RMSE, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_mAC.plot(x,y_mAC, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])

        # if (errors and img_num!=3):
        if (errors):
            np_recons_normalized_masked = np.ma.masked_where(np_labels == 0, np_recons_normalized,copy=True)
            dict_err['NMAE'].append(utils.NMAE(img=np_recons_normalized_masked,ref=np_src))
            dict_err['NRMSE'].append(utils.NRMSE(img=np_recons_normalized_masked,ref=np_src))
            dict_err['PSNR'].append(utils.PSNR(img=np_recons_normalized_masked,ref=np_src))
            dict_err['SSIM'].append(utils.SSIM(img=np_recons_normalized_masked,ref=np_src))
            dict_err['labels'].append(legend[img_num])



    # ax_RC.set_title("Recovery Coefficients for the IEC phantom", fontsize = 18)
    ax_CNR.set_title("Contrast to Noise Ratio for IEC phantom", fontsize = 18)
    ax_RC.legend(fontsize=12)
    ax_CRC.legend(fontsize=12)
    ax_CNR.legend(fontsize=12)
    ax_RMSD.legend(fontsize=12)
    ax_RMSE.legend(fontsize=12)
    # plt.legend()

    if errors:
        colors = ['grey']
        for k in range(len(dict_err['labels'])-1):
            colors.append('black')
        fig_e, ax_e = plt.subplots(2, 2, figsize=(20,20))
        # dict_err['labels'] = [l.split('(') for l in dict_err['labels']]
        ax_e[0,0].bar([k for k in range(len(dict_err['labels']))],dict_err['NMAE'], tick_label = dict_err['labels'], color = colors)
        ax_e[0,0].set_ylabel('NMAE', fontsize = 20, weight="bold")
        ax_e[0,1].bar([k for k in range(len(dict_err['labels']))],dict_err['NRMSE'], tick_label = dict_err['labels'], color = colors)
        ax_e[0,1].set_ylabel('NRMSE', fontsize = 20, weight="bold")
        ax_e[1,0].bar([k for k in range(len(dict_err['labels']))],dict_err['PSNR'], tick_label = dict_err['labels'], color = colors)
        ax_e[1,0].set_ylabel('PSNR', fontsize = 20, weight="bold")
        ax_e[1,1].bar([k for k in range(len(dict_err['labels']))],dict_err['SSIM'], tick_label = dict_err['labels'], color = colors)
        ax_e[1,1].set_ylabel('SSIM', fontsize = 20, weight="bold")
        font_prop={'weight' : 'bold', 'size' : 12, }
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
                print(f"   {lab} : {round(dict_err[f'{errtype}'][i],10)}")
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("images", nargs='+')
    parser.add_argument("--labels")
    parser.add_argument("--labels-json")
    parser.add_argument("--source")
    parser.add_argument("-l", nargs='+')
    parser.add_argument("-c", nargs='+', default = [])
    parser.add_argument("--norm")
    parser.add_argument("--errors",action='store_true')
    parser.add_argument("--auto")
    parser.add_argument("--ref")
    parser.add_argument("--mm",type=int,default=4)
    args = parser.parse_args()


    main()
