#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt
import torch


def make_psf(ks, sx_mm,sy_mm,sz_mm, spacing):
    xk = torch.arange(ks)
    x, y, z = torch.meshgrid(xk, xk, xk, indexing="ij")

    m = (ks - 1) / 2

    sx, sy, sz = sx_mm/spacing, sy_mm/spacing, sz_mm/spacing

    N = np.sqrt(2 * np.pi) ** 3 * sx * sy * sz

    f = 1. / N * torch.exp(-((x - m) ** 2 / sx ** 2 + (y - m) ** 2 / sy ** 2 + (z - m) ** 2 / sz ** 2) / 2)
    return f/f.sum()


def main():
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    fwhm_array = np.linspace(args.fwhm_min, args.fwhm_max, 10)
    sigma_array = fwhm_array / (2 * np.sqrt(2 * np.log(2)))
    nrmse_array = np.zeros_like(sigma_array)

    # rec image
    image = itk.imread(args.image)
    image_array = itk.array_from_image(image)
    spacing = np.array(image.GetSpacing())

    # source image
    source_img = itk.imread(args.source)
    source_array = itk.array_from_image(source_img)
    # normalisation:
    # sum_src = source_array.sum()
    # image_array = image_array / image_array.sum() * sum_src

    image_tensor = torch.Tensor(image_array).to(device)
    source_tensor = torch.Tensor(source_array).to(device)

    kernel_size = 15

    conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=(1, 1, 1),
                           padding=((kernel_size-1)//2,(kernel_size-1)//2,(kernel_size-1)//2), bias=False).to(device)


    for (k,s) in enumerate(sigma_array):
        print(k)
        kernel_tensor=make_psf(ks=kernel_size,sx_mm=s,sy_mm=s,sz_mm=s,spacing=spacing[0]).to(device)
        conv.weight.data = kernel_tensor[None, None, :, :, :]
        with torch.no_grad():
            blurred_source = conv(source_tensor[None,None,:,:,:])

        nrmse = torch.sqrt(torch.sum((image_tensor - blurred_source)**2)) / torch.sqrt(torch.sum(image_tensor**2))
        nrmse_array[k] = nrmse

    fig,ax = plt.subplots()
    # fwhm_array = sigma_array * 2 * np.sqrt(2*np.log(2))
    ax.plot(fwhm_array, nrmse_array)
    ax.set_xlabel("FWHM of filter (mm)")
    ax.set_ylabel("NRMSE")
    plt.show()



    ###############################
    N = args.N
    fwhm_x_array = np.linspace(args.fwhm_min, args.fwhm_max, N)
    fwhm_y_array = np.linspace(args.fwhm_min, args.fwhm_max, N)
    fwhm_z_array = np.linspace(args.fwhm_min, args.fwhm_max, N)
    sigma_x_array = fwhm_x_array / (2 * np.sqrt(2 * np.log(2)))
    sigma_y_array = fwhm_y_array / (2 * np.sqrt(2 * np.log(2)))
    sigma_z_array = fwhm_z_array / (2 * np.sqrt(2 * np.log(2)))
    nrmse_array = np.zeros_like(sigma_y_array)

    min=2024


    ####
    ks = kernel_size
    xk = torch.arange(ks).to(device)
    x, y, z = torch.meshgrid(xk, xk, xk, indexing="ij")
    x,y,z = x.to(device),y.to(device),z.to(device)
    m = (ks - 1) / 2


    for i in range(N):
        print(i)
        for j in range(N):
            for k in range(N):
                sx_mm,sy_mm,sz_mm = sigma_x_array[i], sigma_y_array[j], sigma_z_array[k]
                sx, sy, sz = sx_mm / spacing[0], sy_mm / spacing[0], sz_mm / spacing[0]
                NN = np.sqrt(2 * np.pi) ** 3 * sx * sy * sz
                f = 1. / NN * torch.exp(-((x - m) ** 2 / sx ** 2 + (y - m) ** 2 / sy ** 2 + (z - m) ** 2 / sz ** 2) / 2)
                kernel_tensor = (f / f.sum())
                conv.weight.data = kernel_tensor[None, None, :, :, :]
                with torch.no_grad():
                    blurred_source = conv(source_tensor[None,None,:,:,:])

                nrmse = torch.sqrt(torch.sum((image_tensor - blurred_source)**2)) / torch.sqrt(torch.sum(image_tensor**2))
                nrmse_array[k] = nrmse
                if nrmse < min:
                    lsigma = np.array([sx_mm,sy_mm,sz_mm])
                    min = nrmse
                    print(lsigma, nrmse)

    lfwhm = 2*np.sqrt(2*np.log(2)) * lsigma
    print(f"FWHM : {lfwhm}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--image")
    parser.add_argument("-N", type=int, default = 10)
    parser.add_argument("--fwhm_min", type=float, default = 0)
    parser.add_argument("--fwhm_max", type=float, default = 13)
    args = parser.parse_args()

    main()
