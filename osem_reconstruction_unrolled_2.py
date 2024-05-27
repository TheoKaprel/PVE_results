#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
from itk import RTK as rtk
import itk
import numpy as np

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import get_psf_params

import torch
from DeepPVC.DeepPVC import Model_instance, helpers_data, helpers, helpers_params


def main():
    print(args)

    # data
    projs = itk.imread(args.projs)

    # geom
    nproj,sid = args.nproj,args.sid
    list_angles = np.linspace(0, 360, nproj + 1)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    Offset = projs.GetOrigin()
    for i in range(nproj):
        geometry.AddProjection(sid, 0, list_angles[i], Offset[0], Offset[1])

    # att
    attenuation_image = itk.imread(args.attmap, itk.F)

    # psf
    sigma0_psf, alpha_psf, _ = get_psf_params(machine="siemens-intevo")

    # type
    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]

    option = args.option

    # x0
    like_image = itk.imread(args.like, pixelType)
    constant_image = rtk.ConstantImageSource[imageType].New()
    constant_image.SetSpacing(like_image.GetSpacing())
    constant_image.SetOrigin(like_image.GetOrigin())
    constant_image.SetSize(itk.size(like_image))
    constant_image.SetConstant(1)
    x_n = constant_image.GetOutput()

    multiply_img_filter = itk.MultiplyImageFilter[imageType, imageType, imageType].New()
    divide_img_filter_projs = itk.DivideImageFilter[imageType, imageType, imageType].New()
    divide_img_filter_volume = itk.DivideImageFilter[imageType, imageType, imageType].New()


    ones_projs = rtk.ConstantImageSource[imageType].New()
    ones_projs.SetSpacing(projs.GetSpacing())
    ones_projs.SetOrigin(projs.GetOrigin())
    print(itk.size(projs))
    ones_projs.SetSize(itk.size(projs))
    ones_projs.SetConstant(1)


    back_projector_norm = rtk.ZengBackProjectionImageFilter.New()
    back_projector_norm.SetInput(0, x_n)
    back_projector_norm.SetInput(1, ones_projs.GetOutput())
    back_projector_norm.SetInput(2, attenuation_image)
    back_projector_norm.SetGeometry(geometry)
    back_projector_norm.SetSigmaZero(0)
    back_projector_norm.SetAlpha(0)
    back_projector_norm.Update()
    norm_BP = back_projector_norm.GetOutput()
    itk.imwrite(norm_BP, args.output.replace(".mhd", "_norm.mhd"))

    back_projector_norm_RM = rtk.ZengBackProjectionImageFilter.New()
    back_projector_norm_RM.SetInput(0, x_n)
    back_projector_norm_RM.SetInput(1, ones_projs.GetOutput())
    back_projector_norm_RM.SetInput(2, attenuation_image)
    back_projector_norm_RM.SetGeometry(geometry)
    back_projector_norm_RM.SetSigmaZero(sigma0_psf)
    back_projector_norm_RM.SetAlpha(alpha_psf)
    back_projector_norm_RM.Update()
    norm_BP_RM = back_projector_norm_RM.GetOutput()
    itk.imwrite(norm_BP_RM, args.output.replace(".mhd", "_norm_RM.mhd"))


    if option>0:
        projs_rec_fp_array = itk.array_from_image(itk.imread(args.input_rec_fp))
        device = helpers.get_auto_device(device_mode="cpu")
        pth_file = torch.load(args.pth, map_location=device)
        params = pth_file['params']
        helpers_params.check_params(params)
        params['jean_zay']=False
        model = Model_instance.ModelInstance(params=params, from_pth=args.pth,resume_training=False,device=device)
        model.load_model(pth_path=args.pth)
        model.switch_device(device)
        model.switch_eval()
        model.show_infos()

        # apply PVC(Denoiser(measures))
        projs_np = itk.array_from_image(projs)
        input_projs = (torch.from_numpy(projs_np[None, :, :, :]).to(device),
                       torch.from_numpy(projs_rec_fp_array[None, :, :, :]).to(device))
        with torch.no_grad():
            output_projs = model.forward(input_projs)
        projs_np_PVCNet = output_projs[0, :, :, :].cpu().numpy()

    N=args.n
    for n in range(1,N+1):
        print("iter ", n)

        forward_projector_RM = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector_RM.SetInput(0, projs)
        forward_projector_RM.SetInput(1, x_n)
        forward_projector_RM.SetInput(2, attenuation_image)
        forward_projector_RM.SetGeometry(geometry)
        forward_projector_RM.SetSigmaZero(sigma0_psf)
        forward_projector_RM.SetAlpha(alpha_psf)
        forward_projector_RM.Update()
        x_n_fp = forward_projector_RM.GetOutput()

        x_n_fp_np = itk.array_from_image(x_n_fp)
        projs_np = itk.array_from_image(projs)

        if option>0:
            projs_np = projs_np_PVCNet

            forward_projector_noRM = rtk.ZengForwardProjectionImageFilter.New()
            forward_projector_noRM.SetInput(0, projs)
            forward_projector_noRM.SetInput(1, x_n)
            forward_projector_noRM.SetInput(2, attenuation_image)
            forward_projector_noRM.SetGeometry(geometry)
            forward_projector_noRM.SetSigmaZero(0)
            forward_projector_noRM.SetAlpha(0)
            forward_projector_noRM.Update()
            x_n_fp_noRM = forward_projector_noRM.GetOutput()
            x_n_fp_noRM_np = itk.array_from_image(x_n_fp_noRM)

            if option==1:
                x_n_fp_np = x_n_fp_noRM_np
            elif option==2:
                input_x_n = torch.concat((torch.from_numpy(x_n_fp_np[None,None,:,:,:]),
                                        torch.from_numpy(x_n_fp_noRM_np[None,None,:,:,:])), dim=1).to(device)
                with torch.no_grad():
                    x_n_fp_np = model.UNet_pvc(input_x_n)[0,0,:,:,:].cpu().numpy()
            elif option==3:
                x_n_fp_np = np.random.poisson(lam=x_n_fp_np, size = x_n_fp_np.shape).astype(dtype=np.float32)
                input_x_n = (torch.from_numpy(x_n_fp_np[None, :, :, :]).to(device),
                               torch.from_numpy(x_n_fp_noRM_np[None, :, :, :]).to(device))
                with torch.no_grad():
                    output_x_n = model.forward(input_x_n)
                x_n_fp_np = output_x_n[0, :, :, :].cpu().numpy()


        ratio_np = projs_np / x_n_fp_np
        ratio_np[x_n_fp_np < 1] = 1
        ratio = itk.image_from_array(ratio_np)
        ratio.CopyInformation(projs)


        back_projector = rtk.ZengBackProjectionImageFilter.New()
        back_projector.SetInput(0, x_n)
        back_projector.SetInput(1, ratio)
        back_projector.SetInput(2, attenuation_image)
        back_projector.SetGeometry(geometry)
        if option==0:
            back_projector.SetSigmaZero(sigma0_psf)
            back_projector.SetAlpha(alpha_psf)
        else:
            back_projector.SetSigmaZero(0)
            back_projector.SetAlpha(0)
        ratio_BP = back_projector.GetOutput()


        divide_img_filter_volume.SetInput1(ratio_BP)
        if option==0:
            divide_img_filter_volume.SetInput2(norm_BP_RM)
        else:
            divide_img_filter_volume.SetInput2(norm_BP)

        update = divide_img_filter_volume.GetOutput()

        multiply_img_filter.SetInput1(x_n)
        multiply_img_filter.SetInput2(update)
        x_n = multiply_img_filter.GetOutput()

        itk.imwrite(x_n, args.output.replace('.mhd', f'_{n}.mhd'))
        x_n.DisconnectPipeline()

    itk.imwrite(x_n, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--attmap")
    parser.add_argument("--nproj", type=int)
    parser.add_argument("--sid", type=float)
    parser.add_argument("--like")
    parser.add_argument("--pth")
    parser.add_argument("--input_rec_fp")
    parser.add_argument("--output")
    parser.add_argument("--option", type=int)
    parser.add_argument("-n", type=int)
    args = parser.parse_args()

    main()
