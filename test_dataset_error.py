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

def absRC(imggg, srccc, mask):
    # return (1 - np.abs(1 - (imggg[mask] / srccc[mask]))).mean()
    # return imggg[mask].mean()/srccc[mask].mean()
    # return np.abs(imggg[mask] - srccc[mask] / srccc[mask]).mean()
    # return (imggg[mask] - srccc[mask]).mean()
    return imggg[mask].mean()

def absCRC(imggg, srccc, mask, bg_mask):
    # return (1 - np.abs(1 - (imggg[mask] / srccc[mask]))).mean()
    # return imggg[mask].mean()/srccc[mask].mean()
    # return np.abs(imggg[mask] - srccc[mask] / srccc[mask]).mean()
    # return (imggg[mask] - srccc[mask]).mean()
    # return (imggg[mask].mean()/imggg[bg_mask].mean() - 1)/(srccc[mask].mean()/srccc[bg_mask].mean() - 1)
    # return imggg[mask].mean()

    return (imggg[mask].mean()- imggg[bg_mask].mean())/(srccc[mask].mean() - srccc[bg_mask].mean())


def voxRC(imggg, srccc, mask):
    return imggg[mask]/srccc[mask]

def vaa(imggg, srccc, mask):
    return np.sum(np.abs(srccc[mask] - imggg[mask]) / srccc[mask] < 5 / 100) / np.sum(mask)


def main():
    print(args)
    if args.load is None:
        list_src = glob.glob(os.path.join(args.folder, f"?????_src_4mm.mhd"))
        lMSE_RM, lMSE_PVCNet_sino, lMSE_PVCNet_img, lMSE_PVCNet_sino_img = [], [],[],[]
        lRC_src,lRC_RM, lRC_PVCNet_sino,lRC_PVCNet_img, lRC_PVCNet_sino_img = [], [],[],[],[]
        l_vol = []

        dict_ref_att = np.load(os.path.join(args.folder, "dict_ref_att.npy"), allow_pickle=True).item()

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

            # rm_array = rm_array / rm_array.sum() * src_array.sum()
            # pvcnet_array = pvcnet_array / pvcnet_array.sum() * src_array.sum()
            # pvcnet_img_array = pvcnet_img_array / pvcnet_img_array.sum() * src_array.sum()
            # pvcnet_sino_img_array = pvcnet_sino_img_array / pvcnet_sino_img_array.sum() * src_array.sum()

            lesion_mask_array = itk.array_from_image(lesion_mask_img)
            rois_mask_array = itk.array_from_image(rois_mask_img)

            # lMSE_RM.append(np.sqrt(np.sum((src_array-rm_array)**2))/np.sqrt(np.sum((src_array)**2)))
            # lMSE_PVCNet_sino.append(np.sqrt(np.sum((src_array - pvcnet_array) ** 2)) / np.sqrt(np.sum((src_array) ** 2)))
            # lMSE_PVCNet_img.append(np.sqrt(np.sum((src_array - pvcnet_img_array) ** 2)) / np.sqrt(np.sum((src_array) ** 2)))
            # lMSE_PVCNet_sino_img.append(np.sqrt(np.sum((src_array - pvcnet_sino_img_array) ** 2)) / np.sqrt(np.sum((src_array) ** 2)))

            lMSE_RM.append(vaa(imggg=rm_array, srccc=src_array, mask=rois_mask_array==1))
            lMSE_PVCNet_sino.append(vaa(imggg=pvcnet_array, srccc=src_array, mask=rois_mask_array==1))
            lMSE_PVCNet_img.append(vaa(imggg=pvcnet_img_array, srccc=src_array, mask=rois_mask_array==1))
            lMSE_PVCNet_sino_img.append(vaa(imggg=pvcnet_sino_img_array, srccc=src_array, mask=rois_mask_array==1))

            lesion_mask_array_segmented = skimage.measure.label(lesion_mask_array)
            # lesion_mask_array_segmented = lesion_mask_array

            for lbl in np.unique(lesion_mask_array_segmented):
                if lbl!=0:
                    # lRC_RM.append(rm_array[lesion_mask_array_segmented==lbl].mean()/(src_array[lesion_mask_array_segmented==lbl]).mean())
                    # lRC_PVCNet_sino.append(pvcnet_array[lesion_mask_array_segmented == lbl].mean() / (src_array[lesion_mask_array_segmented == lbl]).mean())
                    # lRC_PVCNet_img.append(pvcnet_img_array[lesion_mask_array_segmented == lbl].mean() / (src_array[lesion_mask_array_segmented == lbl]).mean())
                    # lRC_PVCNet_sino_img.append(pvcnet_sino_img_array[lesion_mask_array_segmented == lbl].mean() / (src_array[lesion_mask_array_segmented == lbl]).mean())

                    # lRC_RM.append(absRC(imggg=rm_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl))
                    # lRC_PVCNet_sino.append(absRC(imggg=pvcnet_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl))
                    # lRC_PVCNet_img.append(absRC(imggg=pvcnet_img_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl))
                    # lRC_PVCNet_sino_img.append(absRC(imggg=pvcnet_sino_img_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl))
                    l_vol.append((lesion_mask_array_segmented==lbl).sum())


                    lRC_src.append(absCRC(imggg=src_array, srccc=src_array, mask=lesion_mask_array_segmented == lbl,bg_mask=rois_mask_array==1))
                    lRC_RM.append(absCRC(imggg=rm_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl,bg_mask=rois_mask_array==1))
                    lRC_PVCNet_sino.append(absCRC(imggg=pvcnet_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl,bg_mask=rois_mask_array==1))
                    lRC_PVCNet_img.append(absCRC(imggg=pvcnet_img_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl,bg_mask=rois_mask_array==1))
                    lRC_PVCNet_sino_img.append(absCRC(imggg=pvcnet_sino_img_array, srccc=src_array, mask=lesion_mask_array_segmented==lbl,bg_mask=rois_mask_array==1))

        # fig,ax  = plt.subplots()
        # ax.plot(lRC_src, lRC_src, color="black")
        # ax.scatter(lRC_src, lRC_RM, color ="blue", s= 5)
        # A = np.vstack([lRC_src, np.ones(len(lRC_src))]).T  # Create matrix with x and constant 1 for intercept
        # coefficients = np.linalg.lstsq(A, lRC_RM, rcond=None)[0]  # Solve for [a, b]
        # a, b = coefficients
        # ax.plot(lRC_src, [a * t + b for t in lRC_src], color = "blue")
        #
        # ax.scatter(lRC_src, lRC_PVCNet_sino, color ="orange", s= 5)
        # A = np.vstack([lRC_src, np.ones(len(lRC_src))]).T  # Create matrix with x and constant 1 for intercept
        # coefficients = np.linalg.lstsq(A, lRC_PVCNet_sino, rcond=None)[0]  # Solve for [a, b]
        # a, b = coefficients
        # ax.plot(lRC_src, [a * t + b for t in lRC_src], color="orange")
        #
        # ax.scatter(lRC_src, lRC_PVCNet_img, color ="red", s= 5)
        # A = np.vstack([lRC_src, np.ones(len(lRC_src))]).T  # Create matrix with x and constant 1 for intercept
        # coefficients = np.linalg.lstsq(A, lRC_PVCNet_img, rcond=None)[0]  # Solve for [a, b]
        # a, b = coefficients
        # ax.plot(lRC_src, [a * t + b for t in lRC_src], color="red")
        #
        # ax.scatter(lRC_src, lRC_PVCNet_sino_img, color ="magenta", s= 5)
        # A = np.vstack([lRC_src, np.ones(len(lRC_src))]).T  # Create matrix with x and constant 1 for intercept
        # coefficients = np.linalg.lstsq(A, lRC_PVCNet_sino_img, rcond=None)[0]  # Solve for [a, b]
        # a, b = coefficients
        # ax.plot(lRC_src, [a * t + b for t in lRC_src], color="magenta")
        #
        # plt.show()

        if args.save is not None:
            # np.save(os.path.join(args.save,"lMSE_RM.npy"),lMSE_RM)
            # np.save(os.path.join(args.save,"lMSE_PVCNet_sino.npy"), lMSE_PVCNet_sino)
            # np.save(os.path.join(args.save,"lMSE_PVCNet_img.npy"), lMSE_PVCNet_img)
            # np.save(os.path.join(args.save,"lMSE_PVCNet_sino_img.npy"), lMSE_PVCNet_sino_img)

            np.save(os.path.join(args.save,"lVAA_RM.npy"),lMSE_RM)
            np.save(os.path.join(args.save,"lVAA_PVCNet_sino.npy"), lMSE_PVCNet_sino)
            np.save(os.path.join(args.save,"lVAA_PVCNet_img.npy"), lMSE_PVCNet_img)
            np.save(os.path.join(args.save,"lVAA_PVCNet_sino_img.npy"), lMSE_PVCNet_sino_img)

            # np.save(os.path.join(args.save,"lRC_RM.npy"),lRC_RM)
            # np.save(os.path.join(args.save,"lRC_PVCNet_sino.npy"), lRC_PVCNet_sino)
            # np.save(os.path.join(args.save,"lRC_PVCNet_img.npy"), lRC_PVCNet_img)
            # np.save(os.path.join(args.save,"lRC_PVCNet_sino_img.npy"), lRC_PVCNet_sino_img)

            np.save(os.path.join(args.save,"labsRC_RM.npy"),lRC_RM)
            np.save(os.path.join(args.save,"labsRC_PVCNet_sino.npy"), lRC_PVCNet_sino)
            np.save(os.path.join(args.save,"labsRC_PVCNet_img.npy"), lRC_PVCNet_img)
            np.save(os.path.join(args.save,"labsRC_PVCNet_sino_img.npy"), lRC_PVCNet_sino_img)
            np.save(os.path.join(args.save,"l_vol.npy"),l_vol)

    else:
        lMSE_RM = np.load(os.path.join(args.load,"lMSE_RM.npy"))
        lMSE_PVCNet_sino = np.load(os.path.join(args.load, "lMSE_PVCNet_sino.npy"))
        lMSE_PVCNet_img = np.load(os.path.join(args.load, "lMSE_PVCNet_img.npy"))
        lMSE_PVCNet_sino_img = np.load(os.path.join(args.load, "lMSE_PVCNet_sino_img.npy"))

        lVAA_RM = np.load(os.path.join(args.load,"lVAA_RM.npy"))
        lVAA_PVCNet_sino = np.load(os.path.join(args.load, "lVAA_PVCNet_sino.npy"))
        lVAA_PVCNet_img = np.load(os.path.join(args.load, "lVAA_PVCNet_img.npy"))
        lVAA_PVCNet_sino_img = np.load(os.path.join(args.load, "lVAA_PVCNet_sino_img.npy"))


        lRC_RM = np.load(os.path.join(args.load,"labsRC_RM.npy"))
        lRC_PVCNet_sino = np.load(os.path.join(args.load, "labsRC_PVCNet_sino.npy"))
        lRC_PVCNet_img = np.load(os.path.join(args.load, "labsRC_PVCNet_img.npy"))
        lRC_PVCNet_sino_img = np.load(os.path.join(args.load, "labsRC_PVCNet_sino_img.npy"))
        l_vol = np.load(os.path.join(args.load,"l_vol.npy"))*(4.7952**3)/1000

    unique_vol = np.unique(l_vol)
    lRC_RM_mean_vol = []
    lRC_PVCNet_mean_vol = []
    lRC_PVCNet_img_mean_vol = []
    lRC_PVCNet_sino_img_mean_vol = []
    for v in unique_vol:
        lRC_RM_mean_vol.append((1-np.abs(lRC_RM[l_vol==v]-1)).mean())
        lRC_PVCNet_mean_vol.append((1 - np.abs(lRC_PVCNet_sino[l_vol == v] - 1)).mean())
        lRC_PVCNet_img_mean_vol.append((1 - np.abs(lRC_PVCNet_img[l_vol == v] - 1)).mean())
        lRC_PVCNet_sino_img_mean_vol.append((1 - np.abs(lRC_PVCNet_sino_img[l_vol == v] - 1)).mean())

    fig,ax = plt.subplots()
    ax.bar(1, np.mean(lMSE_RM), 1, color="blue")
    ax.errorbar(1, np.mean(lMSE_RM), mean_confidence_interval(lMSE_RM), fmt='.', color='black', capsize=10)
    ax.bar(2, np.mean(lMSE_PVCNet_sino), 1, color="orange")
    ax.errorbar(2, np.mean(lMSE_PVCNet_sino), mean_confidence_interval(lMSE_PVCNet_sino), fmt='.', color='black', capsize=10)
    ax.set_xticks([1,2], ['RM', 'PVCNet-sino'])
    ax.set_title("MSE")
    plt.rcParams["figure.figsize"] = (5, 5)

    fig,ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([lMSE_RM, lMSE_PVCNet_sino, lMSE_PVCNet_img, lMSE_PVCNet_sino_img], showmeans=False, showextrema=False, showmedians=False)
    print(parts['bodies'])
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    inds = np.arange(1, 5)
    ax.scatter(inds, [np.mean(lMSE_RM), np.mean(lMSE_PVCNet_sino), np.mean(lMSE_PVCNet_img), np.mean(lMSE_PVCNet_sino_img)], marker='x', color='black', s=10, zorder=3)
    print(np.mean(lMSE_RM), np.mean(lMSE_PVCNet_sino), np.mean(lMSE_PVCNet_img), np.mean(lMSE_PVCNet_sino_img))
    ax.set_title("NRMSE", fontsize = 20)
    set_axis_style(ax, ["RM", "PVCNet-sino", "PVCNet-img", "PVCNet-sino-img"])

    fig,ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([lVAA_RM, lVAA_PVCNet_sino, lVAA_PVCNet_img, lVAA_PVCNet_sino_img], showmeans=False, showextrema=False, showmedians=False)
    print(parts['bodies'])
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    inds = np.arange(1, 5)
    ax.scatter(inds, [np.mean(lVAA_RM), np.mean(lVAA_PVCNet_sino), np.mean(lVAA_PVCNet_img), np.mean(lVAA_PVCNet_sino_img)], marker='x', color='black', s=10, zorder=3)
    ax.set_title("VAA", fontsize = 20)
    set_axis_style(ax, ["RM", "PVCNet-sino", "PVCNet-img", "PVCNet-sino-img"])


    fig,ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([lRC_RM, lRC_PVCNet_sino, lRC_PVCNet_img, lRC_PVCNet_sino_img], showmeans=False, showextrema=False, showmedians=False)
    # parts = ax.violinplot([np.abs(1-lRC_RM), np.abs(1-lRC_PVCNet_sino), np.abs(1-lRC_PVCNet_img), np.abs(1-lRC_PVCNet_sino_img)], showmeans=True, showextrema=False, showmedians=False)
    print(np.mean(lRC_RM), np.mean(lRC_PVCNet_sino), np.mean(lRC_PVCNet_img), np.mean(lRC_PVCNet_sino_img))
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    ax.plot([0.5,1,5], [1,1,1], linestyle="--", color = "black")
    inds = np.arange(1, 5)
    ax.scatter(inds,
               [np.mean(lRC_RM), np.mean(lRC_PVCNet_sino), np.mean(lRC_PVCNet_img), np.mean(lRC_PVCNet_sino_img)],
               marker='x', color='black', s=10, zorder=3)
    ax.set_title("RCs", fontsize = 20)
    set_axis_style(ax, ["RM", "PVCNet-sino", "PVCNet-img", "PVCNet-sino-img"])

    fig,ax = plt.subplots()
    ax.bar(1, np.mean(lRC_RM), 1, color="blue")
    ax.errorbar(1, np.mean(lRC_RM), mean_confidence_interval(lRC_RM), fmt='.', color='black', capsize=10)
    ax.bar(2, np.mean(lRC_PVCNet_sino), 1, color="orange")
    ax.errorbar(2, np.mean(lRC_PVCNet_sino), mean_confidence_interval(lRC_PVCNet_sino), fmt='.', color='black', capsize=10)
    ax.set_xticks([1,2], ['RM', 'PVCNet-sino'])
    ax.set_title("RC")

    plt.rcParams["figure.figsize"] = (15, 6)
    fig,ax = plt.subplots()
    print(len(l_vol))
    print(len(lRC_PVCNet_sino))
    ax.scatter(l_vol, lRC_PVCNet_sino, color="orange", s = 5, alpha = 0.5)
    ax.scatter(l_vol, lRC_RM, color="royalblue", s=5, alpha = 0.5)
    ax.scatter(l_vol, lRC_PVCNet_img, color="lightcoral", s=5, alpha = 0.5)
    ax.scatter(l_vol, lRC_PVCNet_sino_img, color="violet", s=5, alpha = 0.5)
    ax.plot(moving_average(unique_vol), moving_average(lRC_RM_mean_vol),label="RM", color = "blue", linewidth = 3)
    ax.plot(moving_average(unique_vol), moving_average(lRC_PVCNet_mean_vol),label="PVCNet-sino", color = "darkorange", linewidth = 3)
    ax.plot(moving_average(unique_vol), moving_average(lRC_PVCNet_img_mean_vol),label="PVCNet-img", color = "red", linewidth = 3)
    ax.plot(moving_average(unique_vol), moving_average(lRC_PVCNet_sino_img_mean_vol),label="PVCNet-sino-img", color = "magenta", linewidth = 3)
    ax.plot(unique_vol, [1 for _ in unique_vol], '-', color = 'grey')
    ax.set_xlabel("Lesion Volume (mL)", fontsize = 20)
    ax.set_ylabel("Recovery Coefficients (RC)", fontsize = 20)
    ax.set_xlim([-0.5,unique_vol.max()+0.5])
    ax.legend(fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # fig,ax = plt.subplots()
    # ax.plot(moving_average(unique_vol), [b/a for a,b in zip(moving_average(lRC_RM_mean_vol),moving_average(lRC_PVCNet_mean_vol))],label="ratio", color = "grey", linewidth = 3)
    # ax.set_xlabel("Lesion Volume (mL)", fontsize = 20)
    # ax.set_ylabel("Error Gain", fontsize = 20)
    # ax.set_xlim([-0.5, unique_vol.max() + 0.5])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)

    plt.rcParams["savefig.directory"] = "/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset/figs"

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--save")
    parser.add_argument("--load")
    args = parser.parse_args()

    main()
