#!/usr/bin/env python3

import numpy as np


def calc_norm(img,norm):
    if (norm==False or norm==None):
        return 1
    elif norm == "max":
        return img.max()
    elif norm == "sum":
        return img.sum()
    else:
        return img.max()



def normalize(img, norm):
    img_norm = calc_norm(img,norm)
    return img / img_norm


def NRMSE(img,ref):
    rmse = np.sqrt(np.mean((ref - img) ** 2))
    return rmse / np.mean(np.abs(ref))

def NMAE(img,ref):
    mae = np.mean(np.abs(ref - img))
    return mae / np.mean(np.abs(ref))


def PSNR(img, ref):
    max_ref = np.max(ref)
    mse = np.mean((ref - img) ** 2)
    PSNR = 10 * np.log10((max_ref**2)/mse)
    return PSNR

def CNR(mask1, mask2, img):
    mu1 = np.mean(img[mask1])
    mu2 = np.mean(img[mask2])
    std1 = np.std(img[mask1])
    std2 = np.std(img[mask2])
    # CNR = (mu1 - mu2) / (np.sqrt(std1**2 + std2**2))
    CNR = (mu1 - mu2) / std2
    return CNR

def SSIM(img,ref):
    L=np.max(ref)
    k1,k2=0.01,0.03
    c1,c2 = (k1*L)**2, (k2*L)**2
    mu_img=np.mean(img)
    mu_ref=np.mean(ref)
    std_img=np.std(img)
    std_ref=np.std(ref)
    cov_img_ref=covariance(img,ref)

    return (2*mu_img*mu_ref + c1) * (2*cov_img_ref + c2) / ((mu_img**2 + mu_ref**2 + c1)*(std_ref**2 + std_img**2 + c2))

def covariance(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)