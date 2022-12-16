#!/usr/bin/env python3

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth')
@click.option('--network', help = 'ex : unet_denoiser,')
def print_hi(pth, network):
    pth_file = torch.load(pth)

    if network not in pth_file.keys():
        print(f'ERROR: {network} not in {pth_file.keys()} ...!')
        exit(0)

    network_dict = pth_file[network]
    weights = np.array([])

    for layer_weights in network_dict.values():
        weights = np.concatenate((weights, layer_weights.cpu().detach().numpy().ravel()))

    fig,ax = plt.subplots(1,3)
    ax[0].hist(weights, bins= 100)
    ax[1].hist(weights, bins= 100, range=(-1,1))
    ax[2].hist(weights, bins= 1000, range=(-1,1))
    plt.show()

if __name__ == '__main__':
    print_hi()
