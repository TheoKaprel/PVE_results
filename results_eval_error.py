#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=20)
    ax.set_xlim(0.25, len(labels) + 0.75)

def mean_confidence_interval(aa, confidence=0.95):
    a = 1.0 * np.array(aa)
    n = len(a)
    # se = scipy.stats.sem(a)
    #
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    # print(h, 1.96*aa.std(), 1.96 * (aa.std()/np.sqrt(n)))
    h = 1.96 * (aa.std()/np.sqrt(n))
    return h


def main():
    print(args)
    fig,ax =plt.subplots()

    data = []
    lbllls = ["1UNET\nrecfp\n3Blocks",
              "1UNET\nrecfp\n4Blocks",
              "1UNET\nrecfp\n5Blocks",
              "2CNN\nrecfp/PVEloss\n32ch",
              "2CNN\nrecfp/PVEloss\n64ch",
              "2UNETS\nrecfp/PVEloss\n3Blocks",
              "2UNETS\nrecfp/PVEloss\n4Blocks",
              "2UNETS\nrecfp/PVEloss\n5Blocks",
              "2UNETS\nrecfp\n3Blocks",
              "2UNETS\nPVEloss\n3Blocks",
              ]
    for k,inp in enumerate(args.inputs):
        a = np.load(inp)
        data.append(a)
        # print(lbllls[k], a.mean(), a.std(), mean_confidence_interval(a))
        print("---")

    print(len(data))
    ax.set_title("NRMSE")
    ax.violinplot(data,showmeans=True)
    set_axis_style(ax,labels=args.inputs)

    fig,ax =plt.subplots()
    data = []

    # lbllls = args.inputs

    for k,inp in enumerate(args.inputs):
        inp = inp.replace("NRMSE", "NMAE")
        a = np.load(inp)
        data.append(a)
        print(lbllls[k], a.mean(), a.std(), mean_confidence_interval(a), (a.shape))
        print('---')
    parts = ax.violinplot(data,showmeans=True)
    ax.set_ylabel("NMAE", fontsize=20)
    plt.yticks(fontsize=20)

    set_axis_style(ax,labels=lbllls)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(2)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('#D43F3A')
        pc.set_alpha(1)
    plt.rcParams["savefig.directory"] = "/export/home/tkaprelian/Desktop/PVE/Results/output_from_Jean_Zay/jnm_ablation/eval"

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--inputs", nargs="*")
    args = parser.parse_args()

    main()
