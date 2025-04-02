#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt


def main():
    print(args)
    legends = []
    l_mse = []
    l_mae = []
    l_vaa = []
    l_bg_noise = []
    with open(args.input) as file:
        for line in file.readlines():
            line_split = line.split(' ')
            print(line_split)
            if line_split[0] in ['RM', 'iY', 'PVCNet-sino']:
                legends.append(line_split[0])
                l_mse.append(float(line_split[1]))
                l_mae.append(float(line_split[2]))
                l_vaa.append(float(line_split[3]))
                l_bg_noise.append(float(line_split[4][:-2]))


    fs = 15
    fig,ax = plt.subplots(2,2)
    ax = ax.ravel()
    ax[0].bar(legends, l_mse, color=['blue', 'blueviolet', 'gold', 'red', 'orange'], alpha = 0.8)
    # ax[0].bar(legends, l_mse, color=['blue', 'blueviolet', 'orange'])
    ax[0].set_title('NRMSE', fontsize = fs)
    ax[1].bar(legends, l_mae, color=['blue', 'blueviolet', 'gold', 'red', 'orange'], alpha = 0.8)
    # ax[1].bar(legends, l_mae, color=['blue', 'blueviolet', 'orange'])
    ax[1].set_title('NMAE', fontsize = fs)
    ax[2].bar(legends, l_vaa, color=['blue', 'blueviolet', 'gold', 'red', 'orange'], alpha = 0.8)
    # ax[2].bar(legends, l_vaa, color=['blue', 'blueviolet', 'orange'])
    ax[2].set_title('VAA (%)', fontsize = fs)
    ax[3].bar(legends, l_bg_noise, color=['blue', 'blueviolet', 'gold', 'red', 'orange'], alpha = 0.8)
    # ax[3].bar(legends, l_bg_noise, color=['blue', 'blueviolet', 'orange'])
    ax[3].set_title('BG Noise (std)', fontsize = fs)
    for k in range(4):
        ax[k].tick_params(axis='both', which='major', labelsize=fs)
        ax[k].tick_params(axis='both', which='minor', labelsize=fs)

    plt.subplots_adjust(left=0.04, right=0.97, top=0.95, bottom=0.06, hspace=0.25, wspace=0.12)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    main()
