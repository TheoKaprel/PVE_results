#!/usr/bin/env python3

import os
import argparse
import glob
import json
import itk
import numpy as np
import matplotlib.pyplot as plt

from utils import CRC

def get_list_of_iter_img(path_iterations):
    id_d = path_iterations.find("%d")
    list_imgs = glob.glob(f'{path_iterations[:id_d]}*{path_iterations[id_d+2:]}')
    tuple_iter_img = [(int(img[id_d:img.find(path_iterations[id_d+2:])]), img ) for img in list_imgs]
    sorted_tuple_iter_img = sorted(tuple_iter_img, key = lambda iter_img: iter_img[0])
    return sorted_tuple_iter_img

dict_sphereslabels_radius = {
    "iec_sphere_10mm": 10,
    "iec_sphere_13mm": 13,
    "iec_sphere_17mm": 17,
    "iec_sphere_22mm": 22,
    "iec_sphere_28mm": 28,
    "iec_sphere_37mm": 37,
    }


spheres_loc = {1: {'l':41, 'c': [44,45]},
               2: {'l':41, 'c': [29,30,31]},
               3: {'l':28, 'c': [21,22, 23, 24, 25]},
               4: {'l': 16, 'c': np.arange(28,33)},
               5: {'l': 16, 'c': np.arange(41,48)},
               6: {'l':28, 'c': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]}}


def main():
    if (len(args.folders)==1 and args.folders[0]=='auto'):
        args.folders = ['rec_noPVE_noPVC_beta0', 'rec_noPVE_noPVC_beta0.001', 'rec_noPVE_noPVC_beta0.01', 'rec_noPVE_noPVC_beta0.1',
                        'rec_PVE_PVC_beta0', 'rec_PVE_PVC_beta0.1', 'rec_PVE_PVC_beta0.01', 'rec_PVE_PVC_beta0.001', 'rec_DeepPVC_2082981_beta0']
        n_lin, n_col = 4, 3
        args.auto=True
    else:
        n_lin,n_col = len(args.folders), 1
        args.auto = False
    args.slicei=38
    print(args)
    mm = 4
    source_fn = os.path.join(args.volume, f'iec_src_bg_{mm}mm_c_rot_scaled.mhd')
    src = itk.array_from_image(itk.imread(source_fn))

    if args.profiles:
        space = 2
        fig_ps, ax_ps = plt.subplots(n_lin,n_col)
        ax_ps_reshape = ax_ps.reshape(-1)
        for i in range(len(ax_ps_reshape)):
            abs = np.array([0])
            ax_i = ax_ps_reshape[i]
            for sph, loc in spheres_loc.items():
                sph_abs = np.arange(abs[-1]+1,abs[-1]+1+len(loc['c'])+2*space)
                abs = np.concatenate((abs,sph_abs))
                columns = np.concatenate((np.arange(loc['c'][0] - space,loc['c'][0]), loc['c'], np.arange(loc['c'][-1]+1, loc['c'][-1]+1+space)))


                if sph==1:
                    ax_i.plot(sph_abs,src[loc['l'], args.slicei,columns],linestyle='dashed',label = "src",color='black')
                else:
                    ax_i.plot(sph_abs,src[loc['l'], args.slicei,columns],linestyle='dashed',color='black')

    if args.mse:
        fig_mse,ax_mse = plt.subplots()

    if args.rc or args.rciter:
        if args.rc:
            fig_rc, ax_rc = plt.subplots(n_lin, n_col)
        if args.rciter>0:
            fig_rciter, ax_rciter = plt.subplots()
            rciter_param_to_diam = [10, 13, 17, 22, 28, 37]
        labels_fn=os.path.join(args.volume, f'iec_labels_{mm}mm_c_rot.mhd')
        img_labels=itk.array_from_image(itk.imread(labels_fn))
        labels_json=os.path.join(args.volume, f'iec_labels_{mm}mm.json')
        json_labels_file = open(labels_json).read()
        json_labels = json.loads(json_labels_file)
        background_mask = (img_labels > 1)
        for sph in dict_sphereslabels_radius:
            background_mask = background_mask * (img_labels != json_labels[sph])


    for (folder_i,folder) in enumerate(args.folders):
        recons_info_fn = os.path.join(folder, "recons_info.json")
        recons_info = open(recons_info_fn).read()
        print("For folder {}, reconstruction parameters are : ".format(folder))
        print(recons_info)
        recons_info = json.loads(recons_info)


        iter_filename_d = os.path.join(folder,recons_info['iter_filename_d'])
        tuple_iter_img = get_list_of_iter_img(path_iterations=iter_filename_d)

        if args.mse:
            list_iter_plot, list_mse = [],[]
        if args.rciter:
            list_iter_plot_rciter, list_rciter = [],[]

        if args.auto and (args.profiles or args.rc):
            if (recons_info['pve']==False and recons_info['pvc']==False):
                ax_c = 0
            elif (recons_info['pve']==True and recons_info['pvc']==True):
                ax_c = 1
            elif (recons_info['pve']==True and recons_info['pvc']==False and recons_info['deep_pvc']==True):
                ax_c = 2
            if recons_info['beta']==0:
                ax_l = 0
            elif recons_info['beta']==0.001:
                ax_l = 1
            elif recons_info['beta']==0.01:
                ax_l = 2
            elif recons_info['beta']==0.1:
                ax_l = 3
            if args.profiles:
                ax_lc = ax_ps[ax_l,ax_c]
            if args.rc:
                ax_rc_lc = ax_rc[ax_l,ax_c]
        elif args.profiles:
            ax_lc = ax_ps[folder_i]
        for (iter,img_fn) in tuple_iter_img:
            if (iter==1 or iter%args.every==0):
                img = itk.array_from_image(itk.imread(img_fn))

                if args.profiles:
                    abs = np.array([0])
                    for sph, loc in spheres_loc.items():
                        sph_abs = np.arange(abs[-1] + 1, abs[-1] + 1 + len(loc['c']) + 2 * space)
                        abs = np.concatenate((abs, sph_abs))
                        columns = np.concatenate((np.arange(loc['c'][0] - space, loc['c'][0]), loc['c'],
                                                  np.arange(loc['c'][-1] + 1, loc['c'][-1] + 1 + space)))
                        if (folder_i==0 and sph==1):
                            ax_lc.plot(sph_abs, img[loc['l'], args.slicei, columns], label = f"{iter}",
                          alpha = (10+iter)/(10+tuple_iter_img[-1][0]), color = 'blue')
                        else:
                            ax_lc.plot(sph_abs, img[loc['l'], args.slicei, columns],
                                                 alpha=(10 + iter) / (10 + tuple_iter_img[-1][0]), color='blue')

                if args.mse:
                    rmse = np.sqrt(np.mean((img - src)**2))
                    list_iter_plot.append(iter)
                    list_mse.append(rmse)

                if args.rc:
                    list_size, list_rc = [],[]
                    for sph_label in dict_sphereslabels_radius:
                        mask_src= (img_labels == json_labels[sph_label])
                        contrast_recov_coef = CRC(img=img,src=src,mask_src=mask_src,mask_bg=background_mask)
                        list_size.append(dict_sphereslabels_radius[sph_label])
                        list_rc.append(contrast_recov_coef)
                    ax_rc_lc.plot(list_size,list_rc, label=f"{iter}",
                                        alpha=(10 + iter) / (10 + tuple_iter_img[-1][0]), color='blue')

                if args.rciter:
                    sph_label = f'iec_sphere_{rciter_param_to_diam[args.rciter-1]}mm'
                    mask_src= (img_labels == json_labels[sph_label])
                    contrast_recov_coef = CRC(img=img,src=src,mask_src=mask_src,mask_bg=background_mask)
                    list_iter_plot_rciter.append(iter)
                    list_rciter.append(contrast_recov_coef)


        if args.profiles:
            ax_lc.set_title(folder)

        if args.mse:
            ax_mse.plot(list_iter_plot,list_mse,'-o', label = folder, linewidth = 1.5)

        if args.rc:
            ax_rc_lc.legend()
            ax_rc_lc.set_ylim([0,1])
            ax_rc_lc.set_title(folder)

        if args.rciter:
            ax_rciter.plot(list_iter_plot_rciter,list_rciter, '-o', label = folder ,linewidth = 1.5)

    if args.mse:
        ax_mse.legend()
    if args.rciter:
        ax_rciter.legend()

    if args.profiles:
        ax_ps.reshape(-1)[0].legend()

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("folders", nargs='+')
    parser.add_argument("--volume")
    parser.add_argument("--every", type = int, default = 10)
    parser.add_argument("--profiles",action='store_true')
    parser.add_argument("--mse", action='store_true')
    parser.add_argument("--rc", action='store_true')
    parser.add_argument("--rciter", type = int, default = 0,choices = [0,1,2,3,4,5,6])
    args = parser.parse_args()

    main()
