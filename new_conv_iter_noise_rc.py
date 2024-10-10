#!/usr/bin/env python3

import argparse
import os.path

import itk
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import json


def get_MSE_RC_CNR_iters(path_iterations, src,json_labels, np_labels):
    list_imgs,list_iters=[],[]
    for iter in range(1,100):
        file_iter=path_iterations.replace("%d", f'{iter}')
        if os.path.isfile(file_iter):
            list_imgs.append(file_iter)
            list_iters.append(iter)
    if "mlem" not in path_iterations:
        list_iters = [8*i for i in list_iters]

    l_mse = []
    # background mask
    body_mask = (np_labels>0)
    background_mask = (np_labels==1)
    l_RC = []
    l_CNR = []
    src_bg = src[background_mask].mean()
    for iter, img in zip(list_iters, list_imgs):
        imgitk = itk.imread(img)
        imgnp = itk.array_from_image(imgitk)
        imgn = imgnp * 337
        l_mse.append(np.sqrt(np.mean((imgn[body_mask] - src[body_mask]) ** 2)) / np.mean(np.abs(src[body_mask])))

        act_bg = imgn[background_mask].mean()
        CRCs = []
        CNRs = []
        for organ, lbl in json_labels.items():
            if ((organ!="body") and ("label" not in organ)):
                act_organ = imgn[np_labels==lbl].mean()
                src_organ = src[np_labels==lbl].mean()
                CRCs.append((act_organ/act_bg - 1)/(src_organ/src_bg - 1))
                # RCs.append(act_organ/src_organ)
                CNRs.append(np.abs(act_organ - act_bg)/imgn[background_mask].std())

        l_RC.append(sum([1 - abs(crc -1) for crc in CRCs])/len(CRCs))
        l_CNR.append(sum(CNRs)/len(CNRs))


    return l_mse, l_RC, l_CNR, list_iters

def main():
    print(args)

    # open image labels
    img_labels = itk.imread(args.labels)
    np_labels = itk.array_from_image(img_labels)
    # open json labels
    json_labels_filename = args.labels_json
    json_labels_file = open(json_labels_filename).read()
    json_labels = json.loads(json_labels_file)


    srcitk = itk.imread(args.source)
    src = itk.array_from_image(srcitk)
    srcn = src

    fig,ax= plt.subplots(1,2)

    l_mse_RM, l_mRC_RM, l_CNR_RM, list_iters_RM= get_MSE_RC_CNR_iters(args.rm,src = srcn,json_labels=json_labels,np_labels=np_labels)
    l_mse_PVCNet, l_mRC_PVCNet, l_CNR_PVCNet, list_iters_PVCNet = get_MSE_RC_CNR_iters(args.pvcnet,src = srcn,json_labels=json_labels,np_labels=np_labels)
    l_mse_iY, l_mRC_iY, l_CNR_iY, list_iters_iY = get_MSE_RC_CNR_iters(args.iy,src = srcn,json_labels=json_labels,np_labels=np_labels)

    print(l_mse_PVCNet)
    print(l_mse_iY)

    list_iters_PVCNet = list_iters_RM[:10]+[list_iters_RM[9]+lkl for lkl in list_iters_PVCNet]
    list_iters_iY = list_iters_RM+[list_iters_RM[-1]]

    l_mse_PVCNet = l_mse_RM[:10]+l_mse_PVCNet
    l_mRC_PVCNet = l_mRC_RM[:10]+l_mRC_PVCNet
    l_CNR_PVCNet = l_CNR_RM[:10]+l_CNR_PVCNet
    l_mse_iY = l_mse_RM+l_mse_iY
    l_mRC_iY = l_mRC_RM+l_mRC_iY
    l_CNR_iY = l_CNR_RM+l_CNR_iY

    ax[0].plot(list_iters_PVCNet,l_mse_PVCNet,marker='o', linestyle="-", label='PVCNet-sino', color = 'orange', linewidth = 3)
    ax[0].plot(list_iters_iY,l_mse_iY,marker='o', linestyle="-", label='iY', color = 'blueviolet', linewidth = 3)
    ax[0].plot(list_iters_RM,l_mse_RM,marker='o', linestyle="-",label='RM', color = 'blue', linewidth = 3)

    ax[1].plot(l_mRC_PVCNet,l_CNR_PVCNet,marker='o', linestyle="-", color = 'orange', linewidth=3, label="PVCNet-sino")
    ax[1].plot(l_mRC_iY,l_CNR_iY,marker='o', linestyle="-", color = 'blueviolet', linewidth=3, label="iY")
    ax[1].plot(l_mRC_RM,l_CNR_RM,marker='o', linestyle="-", color = 'blue', linewidth=3, label="RM")


    ax[0].legend( fontsize=20)
    ax[0].set_xlabel("Number of Updates", fontsize=20)
    ax[0].set_ylabel("NRMSE", fontsize=20)
    ax[1].legend( fontsize=20)
    ax[1].set_xlabel("mean $CRC_{err}$",  fontsize=20)
    ax[1].set_ylabel("CNR",  fontsize=20)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--rm")
    parser.add_argument("--pvcnet")
    parser.add_argument("--iy")
    parser.add_argument("--labels")
    parser.add_argument("--labels-json")
    args = parser.parse_args()
    main()