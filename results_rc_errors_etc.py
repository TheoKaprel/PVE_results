#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt
import json

import utils



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
    x_rc = np.arange(len(json_labels.keys()))
    width, mult =1/(len(recons_imgs)+1),0

    fig_errors,ax_errors=plt.subplots(2,2)
    axes_err=ax_errors.ravel()
    l_mse,l_mae = [],[]

    fig_histo,ax_histo = plt.subplots()
    hist = np.histogram(np_src.ravel(), bins=100)
    bins = hist[1]
    ax_histo.plot(hist[1][:-1], hist[0],c = 'black', label='ref')

    for img_fn,legend,col in zip(recons_imgs,legends, colors):
        img=itk.imread(img_fn)
        img_array=itk.array_from_image(img)
        if args.norm =="sum":
            img_array_normed = img_array / img_array.sum() * np_src.sum()
        else:
            img_array_normed = img_array

        lrc = []
        lstd=[]
        for organ,lbl in json_labels.items():
            rc = img_array_normed[np_labels==int(lbl)].mean() / np_src[np_labels==int(lbl)].mean()
            lrc.append(rc)
            # lstd.append(img_array_normed[np_labels==int(lbl)].std() / np_src[np_labels==int(lbl)].mean())
            lstd.append((img_array_normed[np_labels==int(lbl)]/np_src[np_labels==int(lbl)]).std())
        offset=width*mult
        ax_RC.bar(x_rc+offset, lrc, width, label=legend, color=col)
        ax_RC.errorbar(x_rc+offset, lrc, lstd,fmt='.', color='black',capsize=10)
        mult+=1

        nmae = utils.NMAE(img=img_array_normed,ref=np_src)
        nrmse = utils.NRMSE(img=img_array_normed, ref=np_src)

        l_mae.append(nmae)
        l_mse.append(nrmse)

        hist= np.histogram(img_array_normed.ravel(), bins = bins)
        ax_histo.plot(hist[1][:-1], hist[0], c = col, label = legend)
        print("+++++++")
        VAA = np.sum(np.abs(np_src[np_src>0]-img_array_normed[np_src>0])/np_src[np_src>0] < 5/100) / np.sum(np_src>0)
        print(f"VAA for {legend} : {VAA*100} %")


    axes_err[0].bar(legends, l_mse, color='grey')
    axes_err[0].set_title('mse')
    axes_err[1].bar(legends, l_mae, color='grey')
    axes_err[1].set_title('mae')

    ax_RC.set_ylabel("RC")
    ax_RC.set_xlabel("labels")
    ax_RC.set_xticks(x_rc+width,[f"{organ}" for organ,lbl in json_labels.items()] )

    ax_RC.axhline(y=1, color='grey', linestyle='--')
    ax_RC.legend(loc='upper left')

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
