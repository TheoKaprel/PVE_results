#!/usr/bin/env python3
import os.path

import click
import itk
from itk import RTK as rtk
import numpy as np

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
@click.option('--like', help = "The output will have the same dimension/sapcing than the like image")
@click.option('--sid', type = float, default = 280, help = "source isocenter to detector distance (mm)")
@click.option('--attenuationmap', '-a')
@click.option('--beta', type = float, default = 0, show_default = True, help = "regularization parameter")
@click.option('--nprojpersubset', type = int, default = 10, show_default = True)
@click.option('-n','--niterations', type = int, default = 5, show_default = True)
@click.option('--FB', 'projector_type', default = "Zeng", show_default = True)
@click.option('--output-every', type = int)
@click.option('--iteration-filename', help = 'If output-every is not null, iteration-filename to output intermediate iterations with %d as a placeholder for iteration number')
@click.option('--iteration-folder')
@click.option('-v', '--verbose', count=True)
def osem_reconstruction_click(input, output,like,sid,attenuationmap,beta, nprojpersubset, niterations, projector_type, output_every, iteration_filename,iteration_folder, verbose):
    osem_reconstruction(input=input, outputfilename=output,like=like,sid=sid,attenuationmap=attenuationmap,
                        beta= beta, nprojpersubset=nprojpersubset, niterations=niterations, projector_type=projector_type, output_every=output_every, iteration_filename=iteration_filename,iteration_folder=iteration_folder, verbose=verbose)

def osem_reconstruction(input, outputfilename,like,sid, attenuationmap,beta, nprojpersubset, niterations, projector_type, output_every, iteration_filename,iteration_folder, verbose):
    if verbose>0:
        print('Begining of reconstruction ...')

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    if verbose>0:
        print('Creating the first output image...')

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
    output_0_ = np.zeros(np.array(itk.size(output_image)))
    output_0 = itk.image_from_array(output_0_).astype(pixelType)
    output_0.CopyInformation(output_image)

    osem.SetInput(1, projections)
    osem.SetGeometry(geometry)
    osem.SetNumberOfIterations(niterations)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)
    osem.SetBetaRegularization(beta)

    if att_corr:
        osem.SetInput(2, attenuation_map)

    if (projector_type=='Zeng'):
        FP = osem.ForwardProjectionType_FP_ZENG
        BP = osem.BackProjectionType_BP_ZENG

        osem.SetSigmaZero(0)
        osem.SetAlphaPSF(0)
    elif projector_type=='Joseph':
        FP = osem.ForwardProjectionType_FP_JOSEPH
        BP = osem.BackProjectionType_BP_JOSEPH
    else:
        print('Wrong projector type')
        exit(0)

    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)

    global iter
    iter = 0
    def callback():
        global iter
        iter+=1
        if (output_every and iter%output_every)==0:
            output_iter = osem.GetOutput()
            itk.imwrite(output_iter, iteration_filename.replace('%d', str(iter)))

        if verbose>0:
            print(f'end of iteration {iter}')

    if output_every:
        if iteration_filename:
            try:
                assert ('%d' in iteration_filename)
            except:
                print(f'Error in iteration filename {iteration_filename}. Should contain a %d to be replaced by the iteration number')
                exit(0)
        elif iteration_folder:
            outputbasename=os.path.basename(outputfilename)
            iteration_filename = os.path.join(iteration_folder,outputbasename.replace('.mh', '_%d.mh'))
        else:
            iteration_filename = outputfilename.replace('.mh', '_%d.mh')
    osem.AddObserver(itk.IterationEvent(), callback)

    if verbose>0:
        print('Reconstruction ...')
    osem.Update()

    # Writer
    if verbose>0:
        print("Writing output image...")
    itk.imwrite(osem.GetOutput(), outputfilename)

    if verbose>0:
        print('Done!')

if __name__ =='__main__':

    osem_reconstruction_click()
