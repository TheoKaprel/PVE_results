#!/usr/bin/env python3

import click
import itk

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--labels')
@click.option('--source')
@click.option('-i', '-recons_img', multiple = True)
def show_RC_curve(labels, source, recons_img):
    print('RC')


if __name__ == '__main__':
    show_RC_curve()
