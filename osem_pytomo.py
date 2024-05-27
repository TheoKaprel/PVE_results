#!/usr/bin/env python3

import argparse
import os
import torch
import itk
import numpy as np


import pytomography
from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM,FilteredBackProjection
from pytomography.projectors import SPECTSystemMatrix
from pytomography.metadata.SPECT import SPECTPSFMeta,SPECTProjMeta,SPECTObjectMeta


def main():
    print(args)

    print("--------projs meta data-------")
    projs=itk.imread(args.projs)
    projs_array=itk.array_from_image(projs).astype(float)
    shape_proj = projs_array.shape
    projs_spacing = np.array(projs.GetSpacing())
    print(projs_spacing)
    dx = projs_spacing[0] / 10
    dz = projs_spacing[1] / 10
    dr = (dx, dx, dz)
    nprojs = args.nprojs
    angles = np.linspace(0, 360, nprojs+1)[:nprojs]
    radii = args.sid * np.ones_like(angles)
    proj_meta = SPECTProjMeta((shape_proj[1], shape_proj[2]), angles, radii)
    proj_meta.filepath = args.projs
    proj_meta.index_peak = 0

    print("--------object meta data-------")
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = SPECTObjectMeta(dr, shape_obj)
    M = np.zeros((4,4))
    M[0] = np.array([dx, 0, 0, 0])
    M[1] = np.array([0, dx, 0, 0])
    M[2] = np.array([0, 0, -dz, 0])
    M[3] = np.array([0, 0, 0, 1])
    object_meta.affine_matrix = M

    print("-------- get projections -------")
    projs_array= projs_array[:240,:,:]
    projections = np.transpose(projs_array[:,::-1,:], (0,2,1)).astype(np.float32)
    photopeak= torch.tensor(projections.copy()).to(pytomography.dtype).to(pytomography.device)
    photopeak = photopeak.unsqueeze(dim=0)

    print("-------- attenuation -------")
    attmap = itk.imread(args.attmap)
    attmap_array = itk.array_from_image(attmap)
    print(attmap_array.shape)
    attmap_array_t = np.transpose(attmap_array[:,::-1,:], (2,0,1)).astype(np.float32)
    t = torch.from_numpy(attmap_array_t)
    print(t.shape)

    tpadded = torch.nn.functional.pad(t, (
    (shape_proj[1] - t.shape[2]) // 2, (shape_proj[1] - t.shape[2]) // 2 + (shape_proj[1] - t.shape[2]) % 2,
    (shape_proj[1] - t.shape[1]) // 2, (shape_proj[1] - t.shape[1]) // 2 + (shape_proj[1] - t.shape[1]) % 2,
    (shape_proj[1] - t.shape[0]) // 2, (shape_proj[1] - t.shape[0]) // 2 + (shape_proj[1] - t.shape[0]) % 2))
    print(tpadded.shape)
    # itk.imwrite(itk.image_from_array(tpadded.cpu().numpy()), os.path.join(args.output_folder, f'att.mhd'))


    if args.algo=="osem":
        psf_meta = SPECTPSFMeta((args.alphapsf, args.sigmazero))
        psf_transform = SPECTPSFTransform(psf_meta)
        att_transform = SPECTAttenuationTransform(attenuation_map=tpadded)

        system_matrix = SPECTSystemMatrix(
            obj2obj_transforms = [att_transform,psf_transform],
            proj2proj_transforms = [],
            object_meta = object_meta,
            proj_meta = proj_meta)

        reconstruction_algorithm = OSEM(
            projections=photopeak,
            system_matrix=system_matrix)
        reconstructed_object = reconstruction_algorithm(n_iters=args.n, n_subsets=args.nprojpersubset)
    elif args.algo=="fbp":
        reconstruction_algorithm = FilteredBackProjection(proj=photopeak,angles=angles)
        reconstructed_object = reconstruction_algorithm()


    print(reconstructed_object.shape)
    output_np = reconstructed_object[0,:,:,:].cpu().numpy()
    itk.imwrite(itk.image_from_array(output_np), args.output)
    #
    # output_np = np.transpose(output_np, (1,2,0))
    # output = itk.image_from_array(output_np)
    #
    # itk.imwrite(output, os.path.join(args.output_folder, f'rec_{args.algo}.mhd'))
    #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--attmap")
    parser.add_argument("--nprojs", type = int)
    parser.add_argument("--sid", type = float)
    parser.add_argument("--algo", type = str)
    parser.add_argument("-n", type = int, default= 5)
    parser.add_argument("--nprojpersubset", type = int, default =15)
    parser.add_argument("--sigmazero", type = float)
    parser.add_argument("--alphapsf", type =float)
    parser.add_argument("--output", '-o')
    args = parser.parse_args()

    main()