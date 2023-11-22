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
        legends = ['src'] + legends
    else:
        legends = list(images)
        legends=['source', 'noPVE-noPVC', 'PVE-RM', 'PVE-DeepPVC', 'PVE-noPVC']

    colors=['black', 'green', 'blue', 'orange', 'red', 'grey', 'blueviolet']
    # colors=['black','blue', 'orange']

    source_array = itk.array_from_image(itk.imread(source))
    print(source_array.shape)

    if slice:
        fig_img,ax_img = plt.subplots(2,2)
        ax_img = ax_img.flatten()

        norm_src = utils.calc_norm(source_array, norm=norm)
        stack_img = [source_array / norm_src]

        for img in (images):
            img_array = itk.array_from_image(itk.imread(img))
            norm_img = utils.calc_norm(img_array, norm=norm)
            stack_img.append(img_array / norm_img)


        # vmin_ = min([np.min(sl[slice,:,:]) for sl in stack_img])
        # vmax_ = max([np.max(sl[slice,:,:]) for sl in stack_img])


        vmin_,vmax_ = 0, 2e-5


        for k in range(len(stack_img)):
            # imsh = ax_img[k].imshow(stack_img[k][slice,:,:], vmin = vmin_, vmax = vmax_)
            imsh = ax_img[k].imshow(stack_img[k][:,slice,:], vmin = vmin_, vmax = vmax_)
            ax_img[k].set_title(legends[k])
            ax_img[k].axis('off')


        fig_img.colorbar(imsh, ax=ax_img)

        plt.suptitle(f'Slice {slice}')

    if profile:
        fig_prof,ax_prof = plt.subplots()
        for k in range(len(stack_img)):
            # ax_prof.plot(stack_img[k][slice,profile,:], '-',marker='.', markersize=5,label = legends[k], color=colors[k], linewidth=1.7)
            ax_prof.plot(stack_img[k][profile,slice,:], '-',marker='.', markersize=5,label = legends[k], color=colors[k], linewidth=1.7)

        ax_prof.legend(fontsize=12)
        ax_prof.set_title("profile",fontsize=18)

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

        ax_mse.bar([k for k in range(len(images))],lrmse, tick_label = legends, color = 'black')
        ax_mse.set_ylabel('NRMSE', fontsize = 20)
        # ax_mse[1].bar([k for k in range(len(images))],lnmae, tick_label = legends[1:], color = 'black')
        # ax_mse[1].set_ylabel('NMAE', fontsize = 20)
        # fig_mse.suptitle(source)


    plt.rcParams["savefig.directory"] = os.getcwd()
    plt.show()



if __name__=='__main__':
    comp_rec_images()