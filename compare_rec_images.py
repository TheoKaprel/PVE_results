#!/usr/bin/env python3

import itk
import numpy as np
import click
import matplotlib.pyplot as plt
import os


import utils



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source', required = True)
@click.option('--img', '-i', 'images', multiple = True, required = True)
@click.option('--legend', '-l', multiple = True)
@click.option('-s','--slice', type = int)
@click.option('-p','--profile', type = int)
@click.option('--mse',  is_flag = True, default = False)
@click.option('--norm')
def comp_rec_images(source,images,legend, slice, profile, mse, norm):
    if legend:
        assert(len(images) == len(legend))
        legends = list(legend)
    else:
        legends = list(images)
        legends=['source', 'noPVE-noPVC', 'PVE-RM', 'PVE-DeepPVC', 'PVE-noPVC']

    colors=['black', 'green', 'blue', 'orange', 'red']

    source_array = itk.array_from_image(itk.imread(source))


    if slice:
        fig_img,ax_img = plt.subplots(1,len(images)+1)

        norm_src = utils.calc_norm(source_array, norm=norm)
        stack_img = [source_array / norm_src]

        for img in (images):
            img_array = itk.array_from_image(itk.imread(img))
            norm_img = utils.calc_norm(img_array, norm=norm)
            stack_img.append(img_array / norm_img)


        vmin_ = min([np.min(sl[slice,:,:]) for sl in stack_img])
        vmax_ = max([np.max(sl[slice,:,:]) for sl in stack_img])


        imsh = ax_img[0].imshow(stack_img[0][slice,:,:], vmin = vmin_, vmax = vmax_)
        ax_img[0].set_title(source)
        for k in range(len(images)):
            imsh = ax_img[k+1].imshow(stack_img[k+1][slice,:,:], vmin = vmin_, vmax = vmax_)
            ax_img[k+1].set_title(legends[k])


        fig_img.colorbar(imsh, ax=ax_img)

        plt.suptitle(f'Slice {slice}')

    if profile:
        fig_prof,ax_prof = plt.subplots()
        ax_prof.plot(stack_img[0][slice,profile, :], '-',marker='.', markersize=5, label=legends[0], color=colors[0], linewidth=1.7)
        for k in range(len(images)):
            ax_prof.plot(stack_img[k+1][slice,profile,:], '-',marker='.', markersize=5,label = legends[k+1], color=colors[k+1], linewidth=1.7)

        ax_prof.legend(fontsize=12)
        ax_prof.set_title("Sphere with 48 mm diameter",fontsize=18)

    if mse:
        fig_mse,ax_mse = plt.subplots()
        lrmse = []
        lnmae = []
        norm_src = utils.calc_norm(source_array, norm=norm)
        src = source_array / norm_src

        for k in range(len(images)):
            img_array = itk.array_from_image(itk.imread(images[k]))
            norm_img = utils.calc_norm(img_array, norm=norm)
            img = img_array/norm_img

            rmse = np.sqrt(np.mean((src - img)**2))
            nrmse = rmse / np.mean(np.abs(src))
            lrmse.append(nrmse)

            mae = np.mean(np.abs(src - img))
            nmae = mae / np.mean(np.abs(src))
            lnmae.append(nmae)

        ax_mse.bar([k for k in range(len(images))],lrmse, tick_label = legends[1:], color = 'black')
        ax_mse.set_ylabel('NRMSE', fontsize = 20)
        # ax_mse[1].bar([k for k in range(len(images))],lnmae, tick_label = legends[1:], color = 'black')
        # ax_mse[1].set_ylabel('NMAE', fontsize = 20)
        # fig_mse.suptitle(source)


    plt.rcParams["savefig.directory"] = os.getcwd()
    plt.show()



if __name__=='__main__':
    comp_rec_images()