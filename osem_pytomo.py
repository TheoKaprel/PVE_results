#!/usr/bin/env python3

import argparse
import os
import time

import numpy as np
from pytomography.io.SPECT import dicom
from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors import SPECTSystemMatrix
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt
import pydicom
import itk

def main():
    print(args)

    # initialize the `path`` variable below to specify the location of the required data
    path_CT = args.ct
    files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
    file_NM = args.projs

    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=1)
    photopeak = dicom.get_projections(file_NM, index_peak=1)
    print(photopeak.shape)

    att_transform = SPECTAttenuationTransform(filepath=files_CT)

    collimator_name = 'SY-LEHR'
    energy_kev = 140
    psf_meta = dicom.get_psfmeta_from_scanner_params(collimator_name, energy_kev)
    psf_transform = SPECTPSFTransform(psf_meta)

    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)

    reconstruction_algorithm = OSEM(
        projections=photopeak,
        system_matrix=system_matrix)

    for n in range(5):
        t0 = time.time()
        print(f'iteration {n} ')
        reconstructed_object = reconstruction_algorithm(n_iters=1, n_subsets=10)
        reconstruction_algorithm = OSEM(projections=photopeak, system_matrix=system_matrix, object_initial=reconstructed_object)
        print(f'took {time.time() - t0}')

    print(reconstructed_object.shape)
    output_np = reconstructed_object[0,:,:,:].cpu().numpy()
    output = itk.image_from_array(output_np)
    itk.imwrite(output, os.path.join(args.outputfolder, 'OSEM_5it_10ss.mhd'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projs")
    parser.add_argument("--ct")
    parser.add_argument("--outputfolder" , '-o')
    args = parser.parse_args()

    main()