import os
import itk
import matplotlib.pyplot as plt
import numpy as np
from itk import RTK as rtk

from PVE_data.Analytical_data.parameters import sigma0pve_default,alphapve_default

# folder=""
inputsrc = "/export/home/tkaprelian/Desktop/PVE/enzeli/PYHTZ.mhd"
N=4


sigma0pve=sigma0pve_default
alphapve=alphapve_default





# projection parameters
spacing,size=4.41806,128
offset = (-spacing * size + spacing) / 2

geometry = rtk.ThreeDCircularProjectionGeometry.New()
geometry.AddProjection(380, 0, 0, offset, offset)

pixelType = itk.F
imageType = itk.Image[pixelType, 3]

source_image = itk.imread(inputsrc, itk.F)
source_array = itk.array_from_image(source_image)
source_array_act = source_array / np.sum(source_array) * float(10**5) * spacing ** 2 / (
            source_image.GetSpacing()[0] ** 3)
source_image_act = itk.image_from_array(source_array_act).astype(itk.F)
source_image_act.SetOrigin(source_image.GetOrigin())
source_image_act.SetSpacing(source_image.GetSpacing())


output_spacing = [spacing, spacing, 1]
offset = (-spacing * size + spacing) / 2
output_offset = [offset, offset,0]

output_image = rtk.ConstantImageSource[imageType].New()
output_image.SetSpacing(output_spacing)
output_image.SetOrigin(output_offset)
output_image.SetSize([size, size, 1])
output_image.SetConstant(0.)

forward_projector = rtk.ZengForwardProjectionImageFilter.New()
forward_projector.SetInput(0, output_image.GetOutput())
forward_projector.SetInput(1, source_image_act)
forward_projector.SetGeometry(geometry)


list_alpha=np.linspace(0,alphapve,N)
list_sigma=np.linspace(0,sigma0pve,N)

fig,ax = plt.subplots(N,N)

for i in range(N):
    for j in range(N):
        alpha = list_alpha[i]
        sigma= list_sigma[j]

        forward_projector.SetSigmaZero(sigma)
        forward_projector.SetAlpha(alpha)
        forward_projector.Update()

        output_forward = forward_projector.GetOutput()
        alpha_fn,sigma_fn=round(alpha,3),round(sigma,3)
        # output_filename = os.path.join(folder, f'{folder}_alpha_{alpha_fn}_sigma_{sigma_fn}.mhd')
        # itk.imwrite(output_forward_PVE, output_filename)

        array_output = itk.array_from_image(output_forward)
        ax[i,j].imshow(array_output[0,:,:])
        ax[i,j].set_title(f'alpha {alpha_fn} / sigma {sigma_fn}')
        print('{}/{}'.format(i,j))

plt.show()


