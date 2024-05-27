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
import utils

def main():
    print(args)
    show_RC_curve(labels=args.labels,
                  labels_json=args.labels_json,
                  source = args.source,
                  recons_img=args.images,
                  legend=args.l,
                  color = args.c,
                  norm=args.norm)


def show_RC_curve(labels, labels_json, source, recons_img, legend,color, norm):
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

    # b = 280
    # FWHM_b = get_FWHM_b(machine="siemens-intevo", b=b)

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
    background_mask = (np_labels==json_labels['background'])

    # open source image
    img_src = itk.imread(source)
    np_src = itk.array_from_image(img_src)
    assert ((np_labels.shape == np_src.shape))

    np_src_norm = utils.calc_norm(np_src,norm)
    np_src_normed = np_src / np_src_norm

    mean_bg_src = np.mean(np_src[background_mask])

    print(f"mean bg src : {mean_bg_src}")


    fig_RC,ax_RC = plt.subplots()
    fig_RC_cf,ax_RC_cf = plt.subplots()
    fig_RC_cf_mask,ax_RC_cf_mask = plt.subplots()
    fig_bg,ax_bg = plt.subplots()
    ax_RC.set_xlabel('Sphere Volume (mL)', fontsize=18)
    ax_RC_cf.set_xlabel('Sphere Volume (mL)', fontsize=18)
    ax_RC_cf_mask.set_xlabel('Sphere Volume (mL)', fontsize=18)
    ax_RC.set_ylabel('RC', fontsize=18)
    ax_RC_cf.set_ylabel('RC with CF (on background)', fontsize=18)
    ax_RC_cf_mask.set_ylabel('RC with bg CF (on mask)', fontsize=18)
    ax_bg.set_ylabel('mean bg activity', fontsize=18)

    plt.rcParams["savefig.directory"] = os.getcwd()
    dict_bg_img = {}
    dict_bg_img["src"] = mean_bg_src

    fig,ax_ratio = plt.subplots()
    ax_ratio.set_title('src:bg ratio')
    dict_sphereslabels_ratio = {}
    for sph_label in dict_sphereslabels_radius:
        mean_act_src = (np_src[np_labels == json_labels[sph_label]]).mean()
        dict_sphereslabels_ratio[sph_label] = mean_act_src / mean_bg_src
    x,y_ratio =[],[]
    for sph_label in dict_sphereslabels_ratio:
        x.append(dict_sphereslabels_radius[sph_label])
        y_ratio.append(dict_sphereslabels_ratio[sph_label])
    ax_ratio.plot(x,y_ratio, label='src')

    fig,ax_volumes = plt.subplots()
    ax_volumes.set_title('effective volume')
    dict_sphereslabels_volumes= {}
    volspacing=np.array(img_src.GetSpacing())
    vox_volum=volspacing[0]*volspacing[1]*volspacing[2]
    for sph_label in dict_sphereslabels_radius:
        dict_sphereslabels_volumes[sph_label] = (np_labels == json_labels[sph_label]).sum() * vox_volum*0.001
    x,y_volumes =[],[]
    for sph_label in dict_sphereslabels_volumes:
        x.append(4/3* np.pi * (dict_sphereslabels_radius[sph_label]/2)**3 * 1/1000)
        y_volumes.append(dict_sphereslabels_volumes[sph_label])
    ax_volumes.plot(x,y_volumes,marker='o', label='src')
    ax_volumes.plot(x,x, label='id', color='black', linestyle=':')




    for img_num,img_file in enumerate(recons_img):
        img_recons = itk.imread(img_file)
        np_recons = itk.array_from_image(img_recons)

        # np_recons_norm = utils.calc_norm(np.ma.masked_where(np_labels==0, np_recons,copy=True), norm)
        np_recons_norm = utils.calc_norm(np_recons, norm)
        np_recons_normalized = np_recons / np_recons_norm

        mean_bg_img = np.mean(np_recons[background_mask])
        dict_bg_img[legend[img_num]] = mean_bg_img
        np_recons_cf = np_recons * mean_bg_src / (mean_bg_img)

        if args.cf_mask is not None:
            cf_mask = itk.array_from_image(itk.imread(args.cf_mask))
            mean_mask_img = np.mean(np_recons[cf_mask==1])
            mean_mask_src = np.mean(np_src[cf_mask==1])
            np_recons_cf_mask = np_recons * mean_mask_src/mean_mask_img
        else:
            np_recons_cf_mask = np_recons

        dict_sphereslabels_RC,dict_sphereslabels_RC_cf,dict_sphereslabels_RC_cf_mask = {},{},{}
        dict_sphereslabels_ratio = {}
        print(img_file)
        fig_hist, ax_hist = plt.subplots(2,4)
        axes_hist = ax_hist.ravel()
        a = 0
        fig_hist.suptitle(img_file)

        for sph_label in dict_sphereslabels_radius:
            mean_act_src = (np_src_normed[np_labels==json_labels[sph_label]]).mean()
            mean_act_img = (np_recons_normalized[np_labels==json_labels[sph_label]]).mean()
            dict_sphereslabels_RC[sph_label] = mean_act_img / mean_act_src

            mean_act_src = (np_src[np_labels==json_labels[sph_label]]).mean()
            mean_act_img = (np_recons_cf[np_labels==json_labels[sph_label]]).mean()
            dict_sphereslabels_RC_cf[sph_label] = mean_act_img / mean_act_src

            mean_act_img = (np_recons[np_labels == json_labels[sph_label]]).mean()
            dict_sphereslabels_ratio[sph_label] = mean_act_img / mean_bg_img

            mean_act_src = (np_src[np_labels==json_labels[sph_label]]).mean()
            mean_act_img = (np_recons_cf_mask[np_labels==json_labels[sph_label]]).mean()
            dict_sphereslabels_RC_cf_mask[sph_label] = mean_act_img / mean_act_src


            axes_hist[a].hist(np_recons_normalized[np_labels==json_labels[sph_label]],bins=100, alpha=0.8)
            axes_hist[a].set_title(sph_label)
            mean_act_src = (np_src_normed[np_labels == json_labels[sph_label]]).mean()
            axes_hist[a].axvline(mean_act_src, linestyle='--', color='black')
            a+=1

        axes_hist[a].hist(np_recons_normalized[np_labels==json_labels['background']], bins=100, alpha=0.8)
        axes_hist[a].set_title("bg")
        mean_act_src = (np_src_normed[np_labels == json_labels["background"]]).mean()
        axes_hist[a].axvline(mean_act_src, linestyle='--', color='black')

        x,y_RC,y_RC_cf,y_RC_cf_mask =[],[],[],[]
        y_ratio=[]
        for sph_label in dict_sphereslabels_RC:
            x.append(4/3* np.pi * (dict_sphereslabels_radius[sph_label]/2)**3 * 1/1000)
            y_RC.append(dict_sphereslabels_RC[sph_label])
            y_RC_cf.append(dict_sphereslabels_RC_cf[sph_label])
            y_RC_cf_mask.append(dict_sphereslabels_RC_cf_mask[sph_label])
            y_ratio.append(dict_sphereslabels_ratio[sph_label])

        ax_RC.plot(x,y_RC, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_RC_cf.plot(x,y_RC_cf, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_RC_cf_mask.plot(x,y_RC_cf_mask, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])
        ax_ratio.plot(x,y_ratio, '-o',markersize = 5, linewidth = 2, color = color[img_num], label = legend[img_num])

        if args.errors:
            NMAE = utils.NMAE(img=np_recons_normalized,ref=np_src_normed)
            NMSE = utils.NRMSE(img=np_recons_normalized,ref=np_src_normed)
            PSNR= utils.PSNR(img=np_recons_normalized,ref=np_src_normed)


        if args.save is not None:
            np.save(os.path.join(args.save, legend[img_num]+'_vol_spheres.npy'), x)
            np.save(os.path.join(args.save, legend[img_num]+'_RC.npy'), y_RC)
            np.save(os.path.join(args.save, legend[img_num]+'_NMAE.npy'), np.array(NMAE))
            np.save(os.path.join(args.save, legend[img_num]+'_NMSE.npy'), np.array(NMSE))
            np.save(os.path.join(args.save, legend[img_num]+'_PSNR.npy'), np.array(PSNR))


    ax_bg.bar(range(len(dict_bg_img)), list(dict_bg_img.values()), tick_label=list(dict_bg_img.keys()))
    ax_RC.plot(x, [1 for _ in x], '--', color = 'grey')
    ax_RC.legend(fontsize=12)
    ax_RC_cf.plot(x, [1 for _ in x], '--', color = 'grey')
    ax_RC_cf.legend(fontsize=12)
    ax_RC_cf_mask.plot(x, [1 for _ in x], '--', color = 'grey')
    ax_RC_cf_mask.legend(fontsize=12)


    if args.save is not None:
        fig_RC.savefig(os.path.join(args.save, "fig_RC.png"))


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
    parser.add_argument("--cf_mask")
    parser.add_argument("--save")
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()
    main()

