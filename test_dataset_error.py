#!/usr/bin/env python3

import argparse
import itk
import matplotlib.pyplot as plt
import scipy
import numpy as np
import glob
import os
import skimage.measure

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average_(a,n=3):
    # 'numpy.cumsum'
    # assert n%2==1
    # cumsum_vec = np.cumsum(np.insert(a, 0, 0))
    # return (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return a

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize = 20)
    ax.set_xlim(0.25, len(labels) + 0.75)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def main():
    print(args)
    if args.load is None:
        list_src = glob.glob(os.path.join(args.folder, f"?????_src_4mm.mhd"))

        lMSE_RM,lMSE_PVCNet = [],[]
        lRC_RM,lRC_PVCNet = [],[]
        l_vol = []

        for i,src_fn in enumerate(list_src):

            ref = src_fn.split(f'_src_4mm.mhd')[0][-5:]
            print(i,ref)
            rm_fn = os.path.join(args.folder,f"{ref}_rec.mhd")
            pvcnet_fn = os.path.join(args.folder,f"{ref}_PVCNet_751113_rec_noRM.mhd")
            lesion_mask_fn = os.path.join(args.folder,f"{ref}_lesion_mask_4mm.mhd")

            src_img = itk.imread(src_fn)
            rm_img = itk.imread(rm_fn)
            pvcnet_img = itk.imread(pvcnet_fn)
            lesion_mask_img = itk.imread(lesion_mask_fn)

            src_array = itk.array_from_image(src_img)
            rm_array = itk.array_from_image(rm_img)
            pvcnet_array = itk.array_from_image(pvcnet_img)
            lesion_mask_array = itk.array_from_image(lesion_mask_img)

            lMSE_RM.append(np.sqrt(np.sum((src_array-rm_array)**2))/np.sqrt(np.sum((src_array)**2)))
            lMSE_PVCNet.append(np.sqrt(np.sum((src_array-pvcnet_array)**2))/np.sqrt(np.sum((src_array)**2)))

            lesion_mask_array_segmented = skimage.measure.label(lesion_mask_array)
            for lbl in np.unique(lesion_mask_array_segmented):
                if lbl!=0:
                    lRC_RM.append(rm_array[lesion_mask_array_segmented==lbl].mean()/(src_array[lesion_mask_array_segmented==lbl]).mean())
                    lRC_PVCNet.append(pvcnet_array[lesion_mask_array_segmented==lbl].mean()/(src_array[lesion_mask_array_segmented==lbl]).mean())
                    l_vol.append((lesion_mask_array_segmented==lbl).sum())

        if args.save is not None:
            np.save(os.path.join(args.save,"lMSE_RM.npy"),lMSE_RM)
            np.save(os.path.join(args.save,"lMSE_PVCNet.npy"),lMSE_PVCNet)
            np.save(os.path.join(args.save,"lRC_RM.npy"),lRC_RM)
            np.save(os.path.join(args.save,"lRC_PVCNet.npy"),lRC_PVCNet)
            np.save(os.path.join(args.save,"l_vol.npy"),l_vol)
    else:
        lMSE_RM = np.load(os.path.join(args.load,"lMSE_RM.npy"))
        lMSE_PVCNet = np.load(os.path.join(args.load,"lMSE_PVCNet.npy"))
        lRC_RM = np.load(os.path.join(args.load,"lRC_RM.npy"))
        lRC_PVCNet = np.load(os.path.join(args.load,"lRC_PVCNet.npy"))
        l_vol = np.load(os.path.join(args.load,"l_vol.npy"))*(4.7952**3)/1000

    unique_vol = np.unique(l_vol)
    lRC_RM_mean_vol = []
    lRC_PVCNet_mean_vol = []
    lRC_RM_mean_vol_ = []
    lRC_PVCNet_mean_vol_ = []
    for v in unique_vol:
        # lRC_RM_mean_vol.append(lRC_RM[l_vol==v].mean())
        lRC_RM_mean_vol.append((1-np.abs(lRC_RM[l_vol==v]-1)).mean())
        # lRC_PVCNet_mean_vol.append(lRC_PVCNet[l_vol==v].mean())
        lRC_PVCNet_mean_vol.append((1-np.abs(lRC_PVCNet[l_vol==v]-1)).mean())

    fig,ax = plt.subplots()
    ax.bar(1, np.mean(lMSE_RM), 1, color="blue")
    ax.errorbar(1, np.mean(lMSE_RM), mean_confidence_interval(lMSE_RM), fmt='.', color='black', capsize=10)
    ax.bar(2, np.mean(lMSE_PVCNet), 1, color="orange")
    ax.errorbar(2, np.mean(lMSE_PVCNet), mean_confidence_interval(lMSE_PVCNet), fmt='.', color='black', capsize=10)
    ax.set_xticks([1,2], ['RM', 'PVCNet-sino'])
    ax.set_title("MSE")
    plt.rcParams["figure.figsize"] = (5, 5)

    fig,ax = plt.subplots()
    parts = ax.violinplot([lMSE_RM, lMSE_PVCNet], showmeans=False, showextrema=False, showmedians=True)
    print(parts['bodies'])
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    inds = np.arange(1, 3)
    ax.scatter(inds, [np.mean(lMSE_RM), np.mean(lMSE_PVCNet)], marker='x', color='black', s=10, zorder=3)
    ax.set_title("NRMSE", fontsize = 20)
    set_axis_style(ax, ["RM", "PVCNet-sino"])

    fig,ax = plt.subplots()
    parts = ax.violinplot([lRC_RM, lRC_PVCNet], showmeans=False, showextrema=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    inds = np.arange(1, 3)
    ax.scatter(inds, [np.mean(lRC_RM), np.mean(lRC_PVCNet)], marker='x', color='black', s=10, zorder=3)
    ax.plot([0.5,1,2.5], [1,1,1], linestyle="--", color = "black")
    ax.set_title("RCs", fontsize = 20)
    set_axis_style(ax, ["RM", "PVCNet-sino"])

    fig,ax = plt.subplots()
    ax.bar(1, np.mean(lRC_RM), 1, color="blue")
    ax.errorbar(1, np.mean(lRC_RM), mean_confidence_interval(lRC_RM), fmt='.', color='black', capsize=10)
    ax.bar(2, np.mean(lRC_PVCNet), 1, color="orange")
    ax.errorbar(2, np.mean(lRC_PVCNet), mean_confidence_interval(lRC_PVCNet), fmt='.', color='black', capsize=10)
    ax.set_xticks([1,2], ['RM', 'PVCNet-sino'])
    ax.set_title("RC")

    plt.rcParams["figure.figsize"] = (15, 6)
    fig,ax = plt.subplots()
    ax.scatter(l_vol, lRC_PVCNet, color="orange", s = 5, alpha = 0.5)
    ax.scatter(l_vol, lRC_RM, color="royalblue", s=5, alpha = 0.5)
    ax.plot(moving_average(unique_vol), moving_average(lRC_RM_mean_vol),label="RM", color = "blue", linewidth = 3)
    # ax.plot(moving_average(unique_vol), moving_average(lRC_RM_mean_vol_),label="RM-abs", color = "blue", linewidth = 3 ,linestyle="dashed")
    ax.plot(moving_average(unique_vol), moving_average(lRC_PVCNet_mean_vol),label="PVCNet-sino", color = "darkorange", linewidth = 3)
    # ax.plot(moving_average(unique_vol), moving_average(lRC_PVCNet_mean_vol_),label="PVCNet-sino-abs", color = "darkorange", linewidth = 3,linestyle="dashed")
    ax.plot(unique_vol, [1 for _ in unique_vol], '-', color = 'grey')
    ax.set_xlabel("Lesion Volume (mL)", fontsize = 20)
    ax.set_ylabel("Recovery Coefficients (RC)", fontsize = 20)
    ax.set_xlim([-0.5,unique_vol.max()+0.5])
    ax.legend(fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    fig,ax = plt.subplots()
    ax.plot(moving_average(unique_vol), [b/a for a,b in zip(moving_average(lRC_RM_mean_vol),moving_average(lRC_PVCNet_mean_vol))],label="ratio", color = "grey", linewidth = 3)
    ax.set_xlabel("Lesion Volume (mL)", fontsize = 20)
    ax.set_ylabel("Error Gain", fontsize = 20)
    ax.set_xlim([-0.5, unique_vol.max() + 0.5])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.rcParams["savefig.directory"] = "/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/figs"

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--save")
    parser.add_argument("--load")
    args = parser.parse_args()

    main()
