#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22nd April, 2024

Finite Difference Stencils

"""
import numpy as np 
import torch 

# Dimensionality,  Number of Points, Order of differentiation, Order of Approximation of the Taylor Series 

def get_stencil(keys):
    dims = keys['dims']
    points = keys['points']
    deriv_order = keys['deriv_order']
    taylor_order = keys['taylor_order']

    if dims == 1:
        if points == 3 and deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
        elif points == 2 and deriv_order == 1 and taylor_order == 2:
            return torch.tensor([
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
    elif dims == 2:
        if points == 5 and deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0., -1., 0.],
                [-1., 4., -1.],
                [0., -1., 0.]
            ], dtype=torch.float32)
        elif points == 9 and deriv_order == 2 and taylor_order == 4:
            return torch.tensor([
                [0, 0, -1/12, 0, 0],
                [0, 0, 4/3, 0, 0],
                [-1/12, 4/3, -5/2, 4/3, -1/12],
                [0, 0, 4/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ], dtype=torch.float32)
        elif points == 15 and deriv_order == 2 and taylor_order == 6:
            return torch.tensor([
                [0, 0, 0, 1/90, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 3/2, 0, 0, 0],
                [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                [0, 0, 0, 3/2, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 1/90, 0, 0, 0]
            ], dtype=torch.float32)

    raise ValueError("Invalid stencil parameters")


