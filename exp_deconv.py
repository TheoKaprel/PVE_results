import matplotlib.pyplot as plt
from skimage import restoration
import itk
import glob
import os

path = "/export/home/tkaprelian/Desktop/PVE/enzeli"
list_projs = glob.glob(f'{path}/?????.mhd')
print(list_projs)


for proj in list_projs:
    fig,ax = plt.subplots(1,4)
    proj_PVE_fn = proj.replace(".mhd", "_PVE.mhd")
    proj_PVfree_fn = proj.replace(".mhd", "_PVfree.mhd")
    proj_PVE = itk.array_from_image(itk.imread(proj_PVE_fn))[0,:,:]
    proj_PVfree = itk.array_from_image(itk.imread(proj_PVfree_fn))[0,:,:]
    psf = restoration.richardson_lucy(proj_PVE, proj_PVfree, filter_epsilon=0.00001)
    ax[0].imshow(proj_PVE)
    ax[1].imshow(proj_PVfree)
    ax[2].imshow(psf, vmin=0, vmax = 1)
    ax[3].plot(psf[63,:])
plt.show()
