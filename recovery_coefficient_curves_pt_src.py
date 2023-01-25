#!/usr/bin/env python3

import click
import itk
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import FWHM_b
import utils
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--dir')
@click.option('--pthref')
@click.option('--norm', is_flag=True, default = False)
@click.option('--errors', is_flag=True, default = False)
def show_RC_curve_pt_src(dir, pthref, norm, errors):
    json_radius_file = open(os.path.join(dir, 'radius.json')).read()
    json_radius = json.loads(json_radius_file)

    list_size, list_RC_noPVE_noPVC, list_RC_PVE_PVC, list_RC_PVE_noPVC= [], [], [], []
    list_RC_DeepPVC = []

    if errors:
        list_NMAE_noPVE_noPVC,list_NRMSE_noPVE_noPVC, list_PSNR_noPVE_noPVC,list_SSIM_noPVE_noPVC=[],[],[],[]
        list_NMAE_PVE_PVC, list_NRMSE_PVE_PVC, list_PSNR_PVE_PVC, list_SSIM_PVE_PVC = [], [], [], []
        list_NMAE_PVE_noPVC, list_NRMSE_PVE_noPVC, list_PSNR_PVE_noPVC, list_SSIM_PVE_noPVC = [], [], [], []
        list_NMAE_DeepPVC, list_NRMSE_DeepPVC, list_PSNR_DeepPVC, list_SSIM_DeepPVC = [], [], [], []
        list_CNR_noPVE_noPVC,list_CNR_PVE_PVC,list_CNR_PVE_noPVC,list_CNR_DeepPVC=[],[],[],[]

    # list_legends=['noPVE-noPVC', 'PVE-PVC', 'PVE-noPVC', f'PVE-DeepPVC-{pthref}']
    list_legends=['noPVE-noPVC', 'PVE-RM', 'PVE-noPVC', f'PVE-DeepPVC']


    for item,(name,radius) in enumerate(json_radius.items()):
        if norm:
            src = itk.array_from_image(itk.imread(os.path.join(dir,name+'.mhd')))
        else:
            src = itk.array_from_image(itk.imread(os.path.join(dir, name+'_scaled.mhd')))
        img_rec_noPVE_noPVC = itk.array_from_image(itk.imread(os.path.join(dir,name+'_rec_noPVE_noPVC.mhd')))
        img_rec_PVE_PVC = itk.array_from_image(itk.imread(os.path.join(dir,name+'_rec_PVE_PVC.mhd')))
        img_rec_PVE_noPVC = itk.array_from_image(itk.imread(os.path.join(dir,name+'_rec_PVE_noPVC.mhd')))
        img_rec_DeepPVC = itk.array_from_image(itk.imread(os.path.join(dir,name+f'_rec_DeepPVC_{pthref}.mhd')))

        inside=(src==np.max(src))
        bgmask=(src==np.min(src[src>0]))

        norm_type="sum" if norm else None
        src = utils.normalize(src,norm=norm_type)
        img_rec_noPVE_noPVC = utils.normalize(img_rec_noPVE_noPVC,norm=norm_type)
        img_rec_PVE_PVC = utils.normalize(img_rec_PVE_PVC,norm=norm_type)
        img_rec_PVE_noPVC = utils.normalize(img_rec_PVE_noPVC,norm=norm_type)
        img_rec_DeepPVC = utils.normalize(img_rec_DeepPVC,norm=norm_type)


        mean_src = np.mean(src[inside])
        mean_noPVE_noPVC = np.mean(img_rec_noPVE_noPVC[inside])
        mean_PVE_PVC = np.mean(img_rec_PVE_PVC[inside])
        mean_PVE_noPVC = np.mean(img_rec_PVE_noPVC[inside])
        mean_DeepPVC = np.mean(img_rec_DeepPVC[inside])

        mean_bg_src = np.mean(src[bgmask])
        mean_bg_noPVE_noPVC = np.mean(img_rec_noPVE_noPVC[bgmask])
        mean_bg_PVE_PVC = np.mean(img_rec_PVE_PVC[bgmask])
        mean_bg_PVE_noPVC = np.mean(img_rec_PVE_noPVC[bgmask])
        mean_bg_DeepPVC = np.mean(img_rec_DeepPVC[bgmask])


        list_size.append(radius *2 / FWHM_b)
        list_RC_noPVE_noPVC.append((mean_noPVE_noPVC - mean_bg_noPVE_noPVC) / (mean_src - mean_bg_src))
        list_RC_PVE_PVC.append((mean_PVE_PVC - mean_bg_PVE_PVC) / (mean_src - mean_bg_src))
        list_RC_PVE_noPVC.append((mean_PVE_noPVC - mean_bg_PVE_noPVC) / (mean_src - mean_bg_src))
        list_RC_DeepPVC.append((mean_DeepPVC - mean_bg_DeepPVC) / (mean_src - mean_bg_src))

        if errors:
            list_NMAE_noPVE_noPVC.append(utils.NMAE(img=img_rec_noPVE_noPVC,ref=src))
            list_NMAE_PVE_PVC.append(utils.NMAE(img=img_rec_PVE_PVC,ref=src))
            list_NMAE_PVE_noPVC.append(utils.NMAE(img=img_rec_PVE_noPVC,ref=src))
            list_NMAE_DeepPVC.append(utils.NMAE(img=img_rec_DeepPVC,ref=src))

            list_NRMSE_noPVE_noPVC.append(utils.NRMSE(img=img_rec_noPVE_noPVC,ref=src))
            list_NRMSE_PVE_PVC.append(utils.NRMSE(img=img_rec_PVE_PVC,ref=src))
            list_NRMSE_PVE_noPVC.append(utils.NRMSE(img=img_rec_PVE_noPVC,ref=src))
            list_NRMSE_DeepPVC.append(utils.NRMSE(img=img_rec_DeepPVC,ref=src))

            list_PSNR_noPVE_noPVC.append(utils.PSNR(img=img_rec_noPVE_noPVC,ref=src))
            list_PSNR_PVE_PVC.append(utils.PSNR(img=img_rec_PVE_PVC,ref=src))
            list_PSNR_PVE_noPVC.append(utils.PSNR(img=img_rec_PVE_noPVC,ref=src))
            list_PSNR_DeepPVC.append(utils.PSNR(img=img_rec_DeepPVC,ref=src))

            list_SSIM_noPVE_noPVC.append(utils.SSIM(img=img_rec_noPVE_noPVC,ref=src,))
            list_SSIM_PVE_PVC.append(utils.SSIM(img=img_rec_PVE_PVC,ref=src))
            list_SSIM_PVE_noPVC.append(utils.SSIM(img=img_rec_PVE_noPVC,ref=src))
            list_SSIM_DeepPVC.append(utils.SSIM(img=img_rec_DeepPVC,ref=src))

            list_CNR_noPVE_noPVC.append(utils.CNR(mask1=inside,mask2=bgmask,img=img_rec_noPVE_noPVC))
            list_CNR_PVE_PVC.append(utils.CNR(mask1=inside,mask2=bgmask,img=img_rec_PVE_PVC))
            list_CNR_PVE_noPVC.append(utils.CNR(mask1=inside,mask2=bgmask,img=img_rec_PVE_noPVC))
            list_CNR_DeepPVC.append(utils.CNR(mask1=inside,mask2=bgmask,img=img_rec_DeepPVC))


    fig,ax = plt.subplots()
    ax.plot(list_size, list_RC_noPVE_noPVC,'-o',markersize = 5,color='green', linewidth = 2, label=list_legends[0])
    ax.plot(list_size, list_RC_PVE_PVC,'-o',markersize = 5,color='blue', linewidth = 2,label=list_legends[1])
    ax.plot(list_size, list_RC_PVE_noPVC,'-o',markersize = 5,color='red', linewidth = 2,label=list_legends[2])
    ax.plot(list_size, list_RC_DeepPVC,'-o',markersize = 5,color='orange', linewidth = 2,label=list_legends[3])
    ax.set_xlabel('Sphere diameter / FWHM', fontsize=18)
    ax.set_ylabel('RC', fontsize=18)
    plt.legend(fontsize=12)

    ax.set_title("Recovery Coefficients for different Sphere Size", fontsize=18)


    if errors:
        fig_NMAE,ax_NMAE=plt.subplots()
        ax_NMAE.plot(list_size,list_NMAE_noPVE_noPVC,'-o', markersize=5,color='green', linewidth=2,label=list_legends[0])
        ax_NMAE.plot(list_size,list_NMAE_PVE_PVC,'-o', markersize=5,color='blue', linewidth=2,label=list_legends[1])
        ax_NMAE.plot(list_size,list_NMAE_PVE_noPVC,'-o', markersize=5,color='red', linewidth=2,label=list_legends[2])
        ax_NMAE.plot(list_size,list_NMAE_DeepPVC,'-o', markersize=5,color='orange', linewidth=2,label=list_legends[3])
        ax_NMAE.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
        ax_NMAE.set_ylabel('NMAE', fontsize=18)
        ax_NMAE.set_title('NMAE')
        ax_NMAE.legend()

        fig_NRMSE,ax_NRMSE=plt.subplots()
        ax_NRMSE.plot(list_size,list_NRMSE_noPVE_noPVC,'-o', markersize=5,color='green', linewidth=2,label=list_legends[0])
        ax_NRMSE.plot(list_size,list_NRMSE_PVE_PVC,'-o', markersize=5,color='blue', linewidth=2,label=list_legends[1])
        ax_NRMSE.plot(list_size,list_NRMSE_PVE_noPVC,'-o', markersize=5,color='red', linewidth=2,label=list_legends[2])
        ax_NRMSE.plot(list_size,list_NRMSE_DeepPVC,'-o', markersize=5,color='orange', linewidth=2,label=list_legends[3])
        ax_NRMSE.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
        ax_NRMSE.set_ylabel('NRMSE', fontsize=18)
        ax_NRMSE.set_title('NRMSE')
        ax_NRMSE.legend()

        fig_PSNR,ax_PSNR=plt.subplots()
        ax_PSNR.plot(list_size,list_PSNR_noPVE_noPVC,'-o', markersize=5,color='green', linewidth=2,label=list_legends[0])
        ax_PSNR.plot(list_size,list_PSNR_PVE_PVC,'-o', markersize=5,color='blue', linewidth=2,label=list_legends[1])
        ax_PSNR.plot(list_size,list_PSNR_PVE_noPVC,'-o', markersize=5,color='red', linewidth=2,label=list_legends[2])
        ax_PSNR.plot(list_size,list_PSNR_DeepPVC,'-o', markersize=5,color='orange', linewidth=2,label=list_legends[3])
        ax_PSNR.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
        ax_PSNR.set_ylabel('PSNR', fontsize=18)
        ax_PSNR.set_title('PSNR')
        ax_PSNR.legend()

        fig_CNR,ax_CNR=plt.subplots()
        ax_CNR.plot(list_size,list_CNR_noPVE_noPVC,'-o', markersize=5,color='green', linewidth=2,label=list_legends[0])
        ax_CNR.plot(list_size,list_CNR_PVE_PVC,'-o', markersize=5,color='blue', linewidth=2,label=list_legends[1])
        ax_CNR.plot(list_size,list_CNR_PVE_noPVC,'-o', markersize=5,color='red', linewidth=2,label=list_legends[2])
        ax_CNR.plot(list_size,list_CNR_DeepPVC,'-o', markersize=5,color='orange', linewidth=2,label=list_legends[3])
        ax_CNR.set_xlabel('Sphere Diameter / FWHM', fontsize=18)
        ax_CNR.set_ylabel('CNR', fontsize=18)
        ax_CNR.set_title('CNR')
        ax_CNR.legend()


    # plt.legend(fontsize=18)
    plt.rcParams["savefig.directory"] = os.getcwd()
    plt.show()



if __name__ == '__main__':
    show_RC_curve_pt_src()
