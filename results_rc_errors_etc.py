#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import utils

from tabulate import tabulate


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def main():
    print(args)

    recons_imgs=args.images
    legends=args.l
    colors=args.c

    if len(legends)>0:
        assert(len(legends) == len(recons_imgs))
    else:
        legends = recons_imgs

    if len(colors)>0:
        assert (len(colors)==len(recons_imgs))
    else:
        colors = ['green', 'blue', 'orange', 'red', 'black', 'magenta', 'cyan','yellow','brown','purple','pink','teal','gold','navy','olive','maroon','gray','lime','indigo','beige','turquoise']


    img_labels = itk.imread(args.labels)
    np_labels = itk.array_from_image(img_labels)
    json_labels_file = open(args.labels_json).read()
    json_labels = json.loads(json_labels_file)

    img_src=itk.imread(args.source)
    np_src = itk.array_from_image(img_src)

    fig_RC,ax_RC = plt.subplots()
    fig_mRC,ax_mRC = plt.subplots()
    x_rc = np.arange(len(json_labels.keys()))
    width, mult =1/(len(recons_imgs)+1),0

    fig_errors,ax_errors=plt.subplots(2,2)
    axes_err=ax_errors.ravel()
    l_mse,l_mae,l_cnr = [],[],[]

    fig_histo,ax_histo = plt.subplots()
    hist = np.histogram(np_src.ravel(), bins=100)
    bins = hist[1]
    ax_histo.plot(hist[1][:-1], hist[0],c = 'black', label='ref')

    tab_rc = []
    tab_ic = []
    tab_errors=[]

    fig_vol, ax_vol = plt.subplots()

    for img_fn,legend,col in zip(recons_imgs,legends, colors):
        img=itk.imread(img_fn)
        img_array=itk.array_from_image(img)
        if args.norm =="sum":
            img_array_normed = img_array / img_array.sum() * np_src.sum()
        elif args.norm=="sum_bg":
            img_array_normed = img_array / img_array[np_labels>0].sum() * np_src[np_labels>0].sum()
        elif args.norm=="bg":
            # mean_bg = img_array[np_labels==1].mean()
            img_array_normed = img_array / img_array[np_labels==1].mean() * np_src[np_labels==1].mean()
        elif args.norm=="CF":
            img_array_normed = img_array * 337
        else:
            img_array_normed = img_array

        lrc = []
        l_mean_rc = []
        lstd=[]
        lstd_=[]
        for organ,lbl in json_labels.items():
            rc = img_array_normed[np_labels==int(lbl)].mean() / np_src[np_labels==int(lbl)].mean()
            mean_rc = (np.abs(img_array_normed-np_src)[np_labels==int(lbl)]).mean()
            lrc.append(rc)
            l_mean_rc.append(mean_rc)
            lll = (img_array_normed[np_labels==int(lbl)]/np_src[np_labels==int(lbl)])
            lstd.append(lll.std() / len(lll))
            lstd_.append(mean_confidence_interval(data=lll))

            if organ!="body":
                volume_lbl = np.sum(np_labels==int(lbl))*(4.7952**3)*1/1000
                ax_vol.scatter(volume_lbl, rc, color=col)
        tab_rc.append([legend]+lrc)
        tab_ic.append([legend]+lstd_)
        offset=width*mult
        ax_RC.bar(x_rc+offset, lrc, width, label=legend, color=col)
        ax_mRC.bar(x_rc+offset, l_mean_rc, width, label=legend, color=col)
        ax_RC.errorbar(x_rc+offset, lrc, lstd,fmt='.', color='black',capsize=10)
        print("lstd ",lstd)
        print("l interv: ",lstd_)
        ax_RC.errorbar(x_rc+offset, lrc, lstd_,fmt='.', color='green',capsize=10)
        mult+=1

        nmae = utils.NMAE(img=img_array_normed[np_labels>=1],ref=np_src[np_labels>=1])
        nrmse = utils.NRMSE(img=img_array_normed[np_labels>=1], ref=np_src[np_labels>=1])
        cnr = utils.CNR(mask1=np_labels>1, mask2=np_labels==1,img=img_array_normed)

        l_mae.append(nmae)
        l_mse.append(nrmse)
        l_cnr.append(cnr)

        hist= np.histogram(img_array_normed.ravel(), bins = bins)
        ax_histo.plot(hist[1][:-1], hist[0], c = col, label = legend)
        print("+++++++")
        VAA = np.sum(np.abs(np_src[np_labels>=1]-img_array_normed[np_labels>=1])/np_src[np_labels>=1] < 5/100) / np.sum(np_labels>=1)
        print(f"VAA for {legend} : {VAA*100} %")

        tab_errors.append([legend] + [nrmse,nmae,VAA*100, cnr])

    print(tabulate(tab_rc,headers=["label"]+list(json_labels.keys())))
    print(tabulate(tab_ic,headers=["label"]+list(json_labels.keys())))
    print(tabulate(tab_errors,headers=["label", "NRMSE", "NMAE", "VAA", "CNR"]))

    axes_err[0].bar(legends, l_mse, color='grey')
    axes_err[0].set_title('mse')
    axes_err[1].bar(legends, l_mae, color='grey')
    axes_err[1].set_title('mae')
    axes_err[2].bar(legends, l_cnr, color='grey')
    axes_err[2].set_title('cnr')
    ax_RC.set_ylabel("RC")
    ax_RC.set_xlabel("labels")
    ax_RC.set_xticks(x_rc+width,[f"{organ}" for organ,lbl in json_labels.items()] )
    ax_RC.axhline(y=1, color='grey', linestyle='--')
    ax_RC.legend(loc='upper left')

    ax_mRC.set_ylabel("mean abs error")
    ax_mRC.set_xlabel("labels")
    ax_mRC.set_xticks(x_rc+width,[f"{organ}" for organ,lbl in json_labels.items()] )
    ax_mRC.axhline(y=1, color='grey', linestyle='--')
    ax_mRC.legend(loc='upper left')

    ax_histo.legend(loc ='upper left')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--source")
    parser.add_argument("--labels-json")
    parser.add_argument("--labels")
    parser.add_argument("-l", nargs='+')
    parser.add_argument("-c", nargs='+', default = [])
    parser.add_argument("--norm")
    parser.add_argument("--save")
    parser.add_argument("--rc", action="store_true")
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()

    main()
