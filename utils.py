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