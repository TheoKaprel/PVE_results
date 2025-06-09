#!/usr/bin/env python3

import argparse
import os.path
import itk
from itk import RTK as rtk
import numpy as np
import glob

def main():
    print(args)

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()

    Offset = [0, 0]
    nproj = args.nproj
    list_angles = np.linspace(0, 360, nproj + 1)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(nproj):
        geometry.AddProjection(args.sid, 0, list_angles[i], Offset[0], Offset[1])
    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(args.niterations)
    osem.SetNumberOfProjectionsPerSubset(args.nprojpersubset)
    osem.SetBetaRegularization(0)

    FP = osem.ForwardProjectionType_FP_JOSEPHATTENUATED
    BP = osem.BackProjectionType_BP_JOSEPHATTENUATED
    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)

    list_projs = glob.glob(os.path.join(args.folder, "?????_PVE_att_noisy.mha"))

    attmap = itk.imread(os.path.join(args.folder, "../data/IEC_BG_attmap_cropped_rot_4mm.mhd"))

    proj_fn = list_projs[args.id]
    ref = proj_fn.split('_')[0][-5:]
    print(ref)

    constant_image = rtk.ConstantImageSource[imageType].New()
    constant_image.SetSpacing(attmap.GetSpacing())
    constant_image.SetOrigin(attmap.GetOrigin())
    constant_image.SetSize(itk.size(attmap))
    constant_image.SetConstant(1)
    output_image = constant_image.GetOutput()
    osem.SetInput(0, output_image)
    projections = itk.imread(proj_fn, pixelType)
    osem.SetInput(1, projections)
    osem.SetInput(2, attmap)
    osem.Update()
    outputfilename = proj_fn.replace("PVE_att_noisy.mha", "rec_5n_8ss.mha")
    itk.imwrite(osem.GetOutput(), outputfilename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--id", type = int)
    parser.add_argument("--sid", type = float)
    parser.add_argument("--nproj", type = int)
    parser.add_argument("--nprojpersubset", type = int)
    parser.add_argument("-n","--niterations", type = int)
    args = parser.parse_args()

    main()
