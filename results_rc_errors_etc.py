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

    bg_labels = itk.array_from_image(itk.imread(args.bg)) if args.bg is not None else None

    img_src=itk.imread(args.source)
    np_src = itk.array_from_image(img_src)

    fig_RC,ax_RC = plt.subplots()
    fig_mRC,ax_mRC = plt.subplots()
    fig_CNR,ax_CNR = plt.subplots()
    fig_vaa, ax_vaa = plt.subplots()
    x_rc = np.arange(len(json_labels.keys()))
    width, mult =1/(len(recons_imgs)+1),0

    fig_errors,ax_errors=plt.subplots(2,2)
    axes_err=ax_errors.ravel()
    l_mse,l_mae,l_cnr,l_psnr = [],[],[],[]
    l_vaa,l_bg_noise = [],[]

    fig_histo,ax_histo = plt.subplots()
    hist = np.histogram(np_src.ravel(), bins=100)
    bins = hist[1]
    ax_histo.plot(hist[1][:-1], hist[0],c = 'black', label='ref')

    tab_rc = []
    tab_ic = []
    tab_MAE = []
    tab_errors=[]

    fig_vol, ax_vol = plt.subplots()
    fig_tbr, ax_tbr = plt.subplots()

    fig_noise,ax_noise = plt.subplots(2,2)
    ax_noise = ax_noise.ravel()
    k=0
    for img_fn,legend,col in zip(recons_imgs,legends, colors):
        k+=1
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
        lcnr = []

        for organ,lbl in json_labels.items():
            rc = img_array_normed[np_labels==int(lbl)].mean() / np_src[np_labels==int(lbl)].mean()
            # rc = img_array_normed[np_labels==int(lbl)].max() / np_src[np_labels==int(lbl)].max()
            mean_rc = (np.abs(img_array_normed-np_src)[np_labels==int(lbl)]).mean()  / np_src[np_labels==int(lbl)].mean()
            # mean_rc = ((img_array_normed-np_src)**2)[np_labels==int(lbl)].mean()
            # mean_rc = 1 - (np.abs(1 - img_array_normed/np_src)[np_labels==int(lbl)]).mean()
            lrc.append(rc)
            l_mean_rc.append(mean_rc)
            lll = (img_array_normed[np_labels==int(lbl)]/np_src[np_labels==int(lbl)])
            lll_ = (np.abs(img_array_normed-np_src)[np_labels==int(lbl)])/ np_src[np_labels==int(lbl)]

            # lstd.append(lll.std())
            a = round(rc,3)
            b = round(lll.std(),4)
            jpp = str(a)+" +- "+str(b)
            print(jpp)
            lstd.append(jpp)
            # lstd.append(img_array_normed[np_labels==int(lbl)].std())
            a = round(mean_rc,2)
            b = round(lll_.std(),2)
            jpp = str(a)+" +- "+str(b)
            lstd_.append(jpp)

            if organ!="body":
                volume_lbl = np.sum(np_labels==int(lbl))*(4.7952**3)*1/1000
                ax_vol.scatter(volume_lbl, mean_rc, color=col)
                ax_tbr.scatter(np_src[np_labels==int(lbl)].mean(), mean_rc, color=col)

                lcnr.append(utils.CNR(mask1=np_labels==int(lbl),mask2=np_labels==1,img = img_array_normed))

        tab_rc.append([legend]+lrc)
        tab_ic.append([legend]+lstd)
        tab_MAE.append([legend]+lstd_)
        offset=width*mult
        ax_RC.bar(x_rc+offset, lrc, width, label=legend, color=col)
        ax_mRC.bar(x_rc+offset, l_mean_rc, width, label=legend, color=col)
        ax_CNR.bar(np.arange(len(x_rc)-1)+offset, lcnr, width, label=legend, color=col)
        # ax_RC.errorbar(x_rc+offset, lrc, lstd,fmt='.', color='black',capsize=10)
        print("lstd ",lstd)
        print("l interv: ",lstd_)
        mult+=1

        nmae = utils.NMAE(img=img_array_normed[np_labels>=1],ref=np_src[np_labels>=1])
        nrmse = utils.NRMSE(img=img_array_normed[np_labels>=1], ref=np_src[np_labels>=1])
        cnr = utils.CNR(mask1=np_labels>1, mask2=np_labels==1,img=img_array_normed)
        psnr = utils.PSNR(img=img_array_normed,ref = np_src)

        l_mae.append(nmae)
        l_mse.append(nrmse)
        l_cnr.append(cnr)
        l_psnr.append(psnr)

        hist= np.histogram(img_array_normed.ravel(), bins = bins)
        ax_histo.plot(hist[1][:-1], hist[0], c = col, label = legend)
        print("+++++++")
        VAA = np.sum(np.abs(np_src[np_labels>=1]-img_array_normed[np_labels>=1])/np_src[np_labels>=1] < 5/100) / np.sum(np_labels>=1)
        print(f"VAA for {legend} : {VAA*100} %")

        lprct = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.6]
        lvaa = []
        for prct in lprct:
            lvaa.append(np.sum(np.abs(np_src[np_labels >= 1] - img_array_normed[np_labels >= 1]) / np_src[
                np_labels >= 1] < prct) / np.sum(np_labels >= 1))
        ax_vaa.scatter(lprct, lvaa, color = col, label=legend)



        ccc = ["blue", "orange", "red", "green", "magenta", "black", "blueviolet", "cyan", 'lightgreen', "darkblue"]
        temp = None
        if bg_labels is not None:
            for bg_lbl in np.unique(bg_labels):
                if bg_lbl!=0:
                    if temp is None:
                        temp = img_array_normed[bg_labels==bg_lbl][None,:]
                    else:
                        temp = np.concatenate((temp,img_array_normed[bg_labels==bg_lbl][None,:]))
                    ax_noise[0].scatter(k, img_array_normed[bg_labels==bg_lbl].mean(), c = ccc[bg_lbl-1])
                    ax_noise[1].scatter(k, img_array_normed[bg_labels==bg_lbl].std(), c = ccc[bg_lbl-1])

            print(k, temp.reshape(-1).std())
            ax_noise[2].scatter(k, temp.reshape(-1).std(), c = "black")
            ax_noise[3].scatter(k,img_array_normed[np_labels==1].std() , c = "black")


        l_vaa.append(VAA*100)
        bg_noise = temp.reshape(-1).std()
        l_bg_noise.append(bg_noise)

        tab_errors.append([legend] + [nrmse, nmae, VAA * 100, cnr, bg_noise])

    print(tabulate(tab_rc,headers=["label"]+list(json_labels.keys())))
    print(tabulate(tab_ic,headers=["label"]+list(json_labels.keys())))
    print(tabulate(tab_MAE,headers=["label"]+list(json_labels.keys())))
    print(tabulate(tab_errors,headers=["label", "NRMSE", "NMAE", "VAA", "CNR", "Bg noise"]))

    ax_vol.set_xscale('log')
    ax_vol.set_title("abs RC en fct du volume")
    ax_tbr.set_xscale('log')
    ax_tbr.set_title("abs RC en fct de l acti dans la src")

    fs = 15
    axes_err[0].bar(legends, l_mse, color=['blue', 'blueviolet', 'gold', 'red', 'orange'])
    axes_err[0].set_title('NRMSE', fontsize = fs)
    axes_err[1].bar(legends, l_mae, color=['blue', 'blueviolet', 'gold', 'red', 'orange'])
    axes_err[1].set_title('NMAE', fontsize = fs)
    axes_err[2].bar(legends, l_vaa, color=['blue', 'blueviolet', 'gold', 'red', 'orange'])
    axes_err[2].set_title('VAA (%)', fontsize = fs)
    axes_err[3].bar(legends, l_bg_noise, color=['blue', 'blueviolet', 'gold', 'red', 'orange'])
    axes_err[3].set_title('BG Noise (std)', fontsize = fs)

    ax_RC.set_ylabel("RC")
    ax_RC.set_xlabel("labels")
    ax_RC.set_xticks(x_rc+width,[f"{organ}" for organ,lbl in json_labels.items()] )
    ax_RC.axhline(y=1, color='grey', linestyle='--')
    ax_RC.legend(loc='upper left')

    ax_vaa.legend()

    ax_mRC.set_ylabel("mean abs error")
    ax_mRC.set_xlabel("labels")
    ax_mRC.set_xticks(x_rc+width,[f"{organ}" for organ,lbl in json_labels.items()] )
    ax_mRC.axhline(y=1, color='grey', linestyle='--')
    ax_mRC.legend(loc='upper left')

    ax_CNR.set_title("CNR")
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
    parser.add_argument("--bg")
    parser.add_argument("--save")
    parser.add_argument("--rc", action="store_true")
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()

    main()
