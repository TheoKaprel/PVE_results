#!/usr/bin/env python3

import argparse
import itk
import matplotlib.pyplot as plt
import scipy
import numpy as np
import glob
import os
import skimage.measure

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize = 20)
    ax.set_xlim(0.25, len(labels) + 0.75)

def dice(img,bg_mask, lesion_mask):
    bg_mean = img[bg_mask].mean()
    max_ = img[lesion_mask].max()
    threshold = 0.5 * (max_ + bg_mean)

    mask_ = np.zeros_like(img)
    mask_[img >= threshold] = 1
    expanded_lesion = scipy.ndimage.binary_dilation(lesion_mask,iterations=3)
    mask = np.logical_and(mask_,expanded_lesion)

    intersection = np.logical_and(mask, lesion_mask)
    dice = 2. * intersection.sum() / (mask.sum() + lesion_mask.sum())
    return dice

def main():
    print(args)
    if args.load is None:
        list_src = glob.glob(os.path.join(args.folder, f"?????_src_4mm.mhd"))
        l_vol = []

        dict_ref_att = np.load(os.path.join(args.folder, "dict_ref_att.npy"), allow_pickle=True).item()

        ldice_rm,ldice_sino,ldice_img,ldice_sino_img = [],[],[],[]

        for i,src_fn in enumerate(list_src):
            ref = src_fn.split(f'_src_4mm.mhd')[0][-5:]
            print(i,ref)
            rm_fn = os.path.join(args.folder,f"{ref}_rec.mhd")
            pvcnet_fn = os.path.join(args.folder,f"{ref}_PVCNet_751113_rec_noRM.mhd")
            pvcnet_img_fn = os.path.join(args.folder,f"{ref}_PVCNet_unet_495390_93_100.npy")
            pvcnet_sino_img_fn = os.path.join(args.folder,f"{ref}_PVCNet_unet_589189_71_100.npy")

            rois_mask_fn = os.path.join("/export/home/tkaprelian/Desktop/PVE/datasets/CTs_validation/data", f"{dict_ref_att[ref]}_rois_labels_cropped_rot_4mm.mhd")

            lesion_mask_fn = os.path.join(args.folder,f"{ref}_lesion_mask_4mm.mhd")

            src_img = itk.imread(src_fn)
            rm_img = itk.imread(rm_fn)
            pvcnet_img = itk.imread(pvcnet_fn)
            lesion_mask_img = itk.imread(lesion_mask_fn)
            rois_mask_img = itk.imread(rois_mask_fn)

            src_array = itk.array_from_image(src_img)
            rm_array = itk.array_from_image(rm_img)
            pvcnet_array = itk.array_from_image(pvcnet_img)
            pvcnet_img_array = np.load(pvcnet_img_fn)
            pvcnet_sino_img_array = np.load(pvcnet_sino_img_fn)

            lesion_mask_array = itk.array_from_image(lesion_mask_img)
            rois_mask_array = itk.array_from_image(rois_mask_img)

            lesion_mask_array_segmented = skimage.measure.label(lesion_mask_array)

            for lbl in np.unique(lesion_mask_array_segmented):
                if lbl!=0:
                    lesion_mask = lesion_mask_array_segmented == lbl

                    ldice_rm.append(dice(img=rm_array,bg_mask=rois_mask_array==1,lesion_mask=lesion_mask))
                    ldice_sino.append(dice(img=pvcnet_array,bg_mask=rois_mask_array==1,lesion_mask=lesion_mask))
                    ldice_img.append(dice(img=pvcnet_img_array,bg_mask=rois_mask_array==1,lesion_mask=lesion_mask))
                    ldice_sino_img.append(dice(img=pvcnet_sino_img_array,bg_mask=rois_mask_array==1,lesion_mask=lesion_mask))

        np.save("/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/dice_rm.npy", ldice_rm)
        np.save("/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/dice_pvcnet_sino.npy", ldice_sino)
        np.save("/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/dice_pvcnet_img.npy", ldice_img)
        np.save("/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/dice_pvcnet_sino_img.npy", ldice_sino_img)
    else:
        ldice_rm =  np.load(os.path.join(args.load, "dice_rm.npy"))
        ldice_sino =  np.load(os.path.join(args.load, "dice_pvcnet_sino.npy"))
        ldice_img =  np.load(os.path.join(args.load, "dice_pvcnet_img.npy"))
        ldice_sino_img =  np.load(os.path.join(args.load, "dice_pvcnet_sino_img.npy"))

    # res = scipy.stats.ks_2samp(data1=ldice_rm,data2 = ldice_sino_img,alternative="greater")
    res_sino = scipy.stats.wilcoxon(x=ldice_rm,y = ldice_sino,alternative = "less")
    res_img = scipy.stats.wilcoxon(x=ldice_rm,y = ldice_img,alternative = "less")
    res_sino_img = scipy.stats.wilcoxon(x=ldice_rm,y = ldice_sino_img,alternative = "less")

    print(f"Stat sino:")
    print(res_sino)
    print(f"Stat img:")
    print(res_img)
    print(f"Stat sino-img:")
    print(res_sino_img)

    fig,ax = plt.subplots()
    ax.hist(ldice_rm,bins=100,color="blue", alpha = 0.5)
    ax.hist(ldice_sino_img,bins=100,color="magenta", alpha = 0.5)

    # shapiro_rm = scipy.stats.shapiro(ldice_rm)
    # shapiro_sino_img = scipy.stats.shapiro(ldice_sino_img)
    # print(f"shapiro rm : {shapiro_rm}")
    # print(f"shapiro sino-img : {shapiro_sino_img}")


    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([ldice_rm, ldice_sino,ldice_img,ldice_sino_img], showmeans=False,showextrema=False, showmedians=False)
    print(parts['bodies'])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    inds = np.arange(1, 5)
    ax.scatter(inds, [np.mean(ldice_rm), np.mean(ldice_sino), np.mean(ldice_img), np.mean(ldice_sino_img)], marker='x', color='black', s=10, zorder=3)
    ax.set_title("Dice", fontsize=20)
    set_axis_style(ax, ["RM", "PVCNet-sino", "PVCNet-img", "PVCNet-sino-img"])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--load")
    args = parser.parse_args()

    main()
