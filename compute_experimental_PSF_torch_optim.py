#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class Blurring_Model(torch.nn.Module):
    def __init__(self, device, spacing):
        super(Blurring_Model, self).__init__()
        self.kernel_size = 9
        self.device = device
        self.spacing = spacing

        xk = torch.arange(self.kernel_size)
        self.x, self.y, self.z = torch.meshgrid(xk, xk, xk, indexing="ij")

        self.m = (self.kernel_size - 1) / 2

        self.sigma_x = torch.nn.Parameter(torch.tensor(2.0))
        self.sigma_y = torch.nn.Parameter(torch.tensor(2.0))
        self.sigma_z = torch.nn.Parameter(torch.tensor(2.0))

        self.padding = (self.kernel_size - 1) // 2

    def make_psf(self):
        sx_mm,sy_mm,sz_mm = self.sigma_x,self.sigma_y,self.sigma_z
        sx, sy, sz = sx_mm/self.spacing[0], sy_mm/self.spacing[1], sz_mm/self.spacing[2]

        N = torch.sqrt(torch.Tensor([2 * np.pi])) ** 3 * sx * sy * sz

        f = 1. / N * torch.exp(-((self.x - self.m) ** 2 / sx ** 2
                                 + (self.y - self.m) ** 2 / sy ** 2
                                 + (self.z - self.m) ** 2 / sz ** 2) / 2)

        return (f / f.sum())[None,None,:,:,:]


    def forward(self, x):
        kernel_weights = self.make_psf().to(x.device)
        return F.conv3d(x, kernel_weights, padding=self.padding,stride=1,groups=x.shape[1])




def main():
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # rec image
    image = itk.imread(args.image)
    image_array = itk.array_from_image(image)
    spacing = np.array(image.GetSpacing())

    # source image
    source_img = itk.imread(args.source)
    source_array = itk.array_from_image(source_img)

    image_tensor = torch.Tensor(image_array).to(device)
    source_tensor = torch.Tensor(source_array).to(device)

    max_source = source_tensor.max()
    source_tensor = source_tensor/ max_source
    image_tensor  = image_tensor / max_source

    model = Blurring_Model(device=device, spacing=spacing)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.1)
    loss_fn = torch.nn.MSELoss()
    loss_list = []
    for epoch in range(args.N):
        optimizer.zero_grad()
        blurred_image = model(source_tensor[None,None,:,:,:])[0,0,:,:,:]
        loss = loss_fn(blurred_image,image_tensor)
        loss.backward()
        optimizer.step()
        print(f"epoch = {epoch}, MSE={loss.item(): .3}")
        loss_list.append(loss.item())

    c = 2*np.sqrt(2 * np.log(2))
    print("FWHMx : ", model.sigma_x *c)
    print("FWHMy : ",model.sigma_y*c)
    print("FWHMz : ",model.sigma_z*c)
    fig,ax  = plt.subplots()
    ax.plot(range(args.N), loss_list)

    fig, ax = plt.subplots(3,3)
    ax_ = ax.ravel()
    kernel = model.make_psf().detach().numpy()
    for k in range(9):
        ax_[k].imshow(kernel[0,0,k,:,:], vmin = 0, vmax = kernel.max())
        ax_[k].set_title(str(k))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--image")
    parser.add_argument("-N", type=int, default = 10)
    args = parser.parse_args()

    main()
