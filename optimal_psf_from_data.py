#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
from itk import RTK as rtk
import numpy as np
import itk
import time

def main():
    print(args)

    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]

    print("reading geom...")
    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(args.geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    print("reading projs...")
    input_projs = itk.imread(args.projs)
    input_projs_array = itk.array_from_image(input_projs)
    spacing_proj = np.array(input_projs.GetSpacing())[0]
    size_proj =input_projs_array.shape[1]
    nproj =input_projs_array.shape[0]
    print(f"{nproj} projs found")

    print("reading attmap...")
    attmap = itk.imread(args.attmap, pixelType)

    print("reading src...")
    src = itk.imread(args.src,pixelType)



    output_spacing = [spacing_proj,spacing_proj, 1]
    offset = (-spacing_proj * size_proj + spacing_proj) / 2
    output_offset = [offset, offset, (-nproj+1)/2]
    output_proj = rtk.ConstantImageSource[imageType].New()
    output_proj.SetSpacing(output_spacing)
    output_proj.SetOrigin(output_offset)
    output_proj.SetSize([size_proj, size_proj, nproj])
    output_proj.SetConstant(0.)
    output_proj.Update()
    print("projector creation...")

    forward_projector = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector.SetInput(0, output_proj.GetOutput())
    forward_projector.SetGeometry(geometry)
    forward_projector.SetInput(2, attmap)
    forward_projector.SetInput(1, src)

    N = 10
    # list_alpha =np.linspace(0.03*0.8,0.03*1.2, 8)
    list_alpha =np.linspace(0.03*0.8,0.03*1.2, 16)
    list_sigma= np.linspace(1,2.5, 16)
    # list_sigma= np.array([2.5])

    list_errs = np.zeros((list_sigma.shape[0], list_alpha.shape[0]))
    for i,alpha in enumerate(list_alpha):
        forward_projector.SetAlpha(alpha)
        for j,sigma in enumerate(list_sigma):
            print(sigma, alpha)
            t0 = time.time()

            forward_projector.SetSigmaZero(sigma)

            print("projecting...")
            forward_projector.Update()
            output_forward_PVE = forward_projector.GetOutput()
            output_forward_PVE.DisconnectPipeline()
            output_array = itk.array_from_image(output_forward_PVE)

            MSE = np.mean(np.sum(output_array-input_projs_array)**2)/np.mean(np.sum(output_array)**2)
            list_errs[j,i] = MSE
            print(f"ok ({time.time()-t0} s)")

    fig,ax =plt.subplots()
    for j in range(list_sigma.shape[0]):
        ax.plot(list_alpha, list_errs[j,:], label=f'{list_sigma[j]}')

    np.save("/export/home/tkaprelian/Desktop/PVE/datasets/INTEVO_IEC/INTEVO_LU/B22/psf_determination/errors.npy", list_errs)
    np.save("/export/home/tkaprelian/Desktop/PVE/datasets/INTEVO_IEC/INTEVO_LU/B22/psf_determination/alphas.npy", list_alpha)
    np.save("/export/home/tkaprelian/Desktop/PVE/datasets/INTEVO_IEC/INTEVO_LU/B22/psf_determination/sigmas.npy", list_sigma)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--projs")
    parser.add_argument("--out")
    parser.add_argument("--geom")
    parser.add_argument("--attmap")
    parser.add_argument("--sigmazero", type=float)
    parser.add_argument("--alphapsf", type=float)
    args = parser.parse_args()

    main()
