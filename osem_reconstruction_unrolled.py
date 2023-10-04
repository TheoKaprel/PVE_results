#!/usr/bin/env python3

import click
import itk
from itk import RTK as rtk
import numpy as np

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from PVE_data.Analytical_data.parameters import get_psf_params

def strParamToArray(str_param):
    array_param = np.array(str_param.split(','))
    array_param = array_param.astype(np.float)
    if len(array_param) == 1:
        array_param = np.array([array_param[0].astype(np.float)] * 3)
    return array_param[::-1]


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', '-i', help = 'input projections')
@click.option('--output', '-o', help = 'Output filename of desired type (mhd/mha)')
@click.option('--start')
@click.option('--like')
@click.option('--size', type = str)
@click.option('--spacing', type = str)
@click.option('--geom', '-g')
@click.option('--sid', type = float)
@click.option('--attenuationmap', '-a')
@click.option('--beta', type = float, default = 0, show_default = True)
@click.option('--pvc', is_flag = True, default = False, help = 'if --pvc, resolution correction')
@click.option('--spect_system',type = str, default = "ge-discovery")
@click.option('--nprojpersubset', type = int, default = 10, show_default = True)
@click.option('-n','--niterations', type = int, default = 5, show_default = True)
@click.option('--FB', 'projector_type', default = "Zeng", show_default = True)
@click.option('--regularization', '-r')
@click.option('--output-every', type = int)
@click.option('--iteration-filename', help = 'If output-every is not null, iteration-filename to output intermediate iterations with %d as a placeholder for iteration number')
@click.option('-v', '--verbose', count=True)
def osem_reconstruction_click(input,start, output,like,size,spacing, geom, sid,attenuationmap,beta, pvc,spect_system, nprojpersubset, niterations, projector_type,regularization, output_every, iteration_filename, verbose):
    osem_reconstruction(input=input,start=start, outputfilename=output,like=like,size=size,spacing=spacing, geom=geom,sid=sid,attenuationmap=attenuationmap,
                        beta= beta, pvc=pvc,spect_system=spect_system, nprojpersubset=nprojpersubset, niterations=niterations, projector_type=projector_type,regularization=regularization, output_every=output_every, iteration_filename=iteration_filename, verbose=verbose)

def osem_reconstruction(input,start, outputfilename,like,size,spacing, geom,sid, attenuationmap,beta, pvc,spect_system, nprojpersubset, niterations, projector_type,regularization, output_every, iteration_filename, verbose):
    if verbose>0:
        print('Begining of reconstruction ...')

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    if verbose>0:
        print('Creating the first output image...')
    if start:
        output_image = itk.imread(start, pixelType)
    elif (size and spacing):

        vSize = strParamToArray(size).astype(int)
        vSpacing = strParamToArray(spacing)
        vOffset = [(-sp__*size__ + sp__)/2 for (sp__,size__) in zip(vSpacing,vSize)]
        output_array = np.ones(vSize)
        output_image = itk.image_from_array(output_array)
        output_image.SetSpacing(vSpacing)
        output_image.SetOrigin(vOffset)
        output_image = output_image.astype(pixelType)
    else:
        like_image = itk.imread(like, pixelType)
        constant_image = rtk.ConstantImageSource[imageType].New()
        constant_image.SetSpacing(like_image.GetSpacing())
        constant_image.SetOrigin(like_image.GetOrigin())
        constant_image.SetSize(itk.size(like_image))
        constant_image.SetConstant(1)
        output_image = constant_image.GetOutput()
    if verbose>0:
        print('Reading input projections...')
    projections = itk.imread(input, pixelType)
    nproj = itk.size(projections)[2]
    if verbose>0:
        print(f'{nproj} projections')

    if geom:
        print('Reading geometry file ...')
        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(geom)
        xmlReader.GenerateOutputInformation()
        geometry = xmlReader.GetOutputObject()
        if verbose>0:
            print(geom + ' is open!')
    elif sid:
        print(f'Creating geometry file : nprojs = {nproj} / sid = {sid}')

        if projector_type=="Zeng":
            Offset = projections.GetOrigin()
        else:
            Offset = [0,0]

        list_angles = np.linspace(0,360,nproj+1)
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for i in range(nproj):
            geometry.AddProjection(sid, 0, list_angles[i], Offset[0], Offset[1])
        if verbose>0:
            print(f'Created geom file with {nproj} angles and Offset = {Offset[0]},{Offset[1]}')

    if verbose>0:
        print('Reading attenuation map ...')
    if (attenuationmap):
        attmap_filename = attenuationmap
        attenuation_map = itk.imread(attmap_filename, pixelType)
        att_corr = True
    else:
        att_corr = False
        if verbose>0:
            print('no att map but ok')

    if verbose>0:
        print('Set OSEM parameters ...')
    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, output_image)
    osem.SetInput(1, projections)

    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(1)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)

    osem.SetBetaRegularization(beta)

    if att_corr:
        osem.SetInput(2, attenuation_map)

    sigma0_psf, alpha_psf = get_psf_params(machine=spect_system)

    if (projector_type=='Zeng'):
        FP = osem.ForwardProjectionType_FP_ZENG
        BP = osem.BackProjectionType_BP_ZENG

        if pvc:

            osem.SetSigmaZero(sigma0_psf)
            osem.SetAlpha(alpha_psf)
        else:
            osem.SetSigmaZero(0)
            osem.SetAlpha(0)
    elif projector_type=='Joseph':
        FP = osem.ForwardProjectionType_FP_JOSEPH
        BP = osem.BackProjectionType_BP_JOSEPH

    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)

    # normalisation image
    # ones_projections_np = np.ones(np.array(itk.size(projections)))
    ones_projections_np = np.ones(itk.array_from_image(projections).shape)
    ones_projections = itk.image_from_array(ones_projections_np).astype(pixelType)
    ones_projections.CopyInformation(projections)

    output_0_np = np.zeros(itk.array_from_image(output_image).shape)
    output_0 = itk.image_from_array(output_0_np).astype(pixelType)
    output_0.CopyInformation(output_image)

    back_projector = rtk.ZengBackProjectionImageFilter.New()
    back_projector.SetInput(0, output_0)
    back_projector.SetInput(1, ones_projections)
    back_projector.SetGeometry(geometry)
    back_projector.SetSigmaZero(sigma0_psf)
    back_projector.SetAlpha(alpha_psf)
    back_projector.Update()
    output_normalization_volume = back_projector.GetOutput()
    output_normalization_volume.DisconnectPipeline()
    # itk.imwrite(output_normalization_volume, 'norm.mhd')
    output_normalization_volume_np = itk.array_from_image(output_normalization_volume)

    # Regularization image
    regul_img = itk.imread(regularization)
    regul_np = itk.array_from_image(regul_img)
    gamma = 0.01
    delta = 1./(gamma * output_normalization_volume_np)

    if verbose>0:
        print('Reconstruction ...')

    global iter
    iter = 1
    while iter <= niterations:

        osem.Update()
        output_image = osem.GetOutput()
        output_image.DisconnectPipeline()

        # REGULARIZATION
        output_image_EM = itk.array_from_image(output_image)
        output_regularized_np = (2 * output_image_EM) / ( (1 - delta * regul_np) + np.sqrt((1 - delta * regul_np)**2 + 4 * delta * output_image_EM))
        output_regularized = itk.image_from_array(output_regularized_np)
        output_regularized.CopyInformation(output_image)

        osem.SetInput(0, output_regularized)

        itk.imwrite(output_regularized, iteration_filename.replace('%d', str(iter)))
        if verbose>0:
            print(f'end of iteration {iter}')
        iter += 1

    # Writer
    if verbose>0:
        print("Writing output image...")
    itk.imwrite(output_image, outputfilename)

    if verbose>0:
        print('Done!')





if __name__ =='__main__':

    osem_reconstruction_click()
