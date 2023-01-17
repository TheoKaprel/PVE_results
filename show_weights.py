#!/usr/bin/env python3

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--pth')
@click.option('--network', help = 'ex : unet_denoiser,unet_pvc ...')
@click.option('--layer')
def print_hi(pth, network, layer):
    pth_file = torch.load(pth)

    if network not in pth_file.keys():
        print(f'ERROR: {network} not in {pth_file.keys()} ...!')
        exit(0)

    network_dict = pth_file[network]
    weights = np.array([])
    print(network_dict.keys())

    if layer==None:
        values = network_dict.values()
    else:
        values = network_dict[layer]
        print(values.shape)

        for _ in range(3):
            i_ = np.random.randint(0,values.shape[0])
            for __ in range(3):
                i__ = np.random.randint(0, values.shape[1])
                fig,ax = plt.subplots()
                ax.imshow(values[i_,i__,:,:].detach().cpu().numpy())
        plt.show()



    for layer_weights in values:
        weights = np.concatenate((weights, layer_weights.cpu().detach().numpy().ravel()))

    fig,ax = plt.subplots(1,3)
    ax[0].hist(weights, bins= 100)
    ax[1].hist(weights, bins= 100, range=(-1,1))
    ax[2].hist(weights, bins= 1000, range=(-1,1))




    plt.show()

if __name__ == '__main__':
    print_hi()
