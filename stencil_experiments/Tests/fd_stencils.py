#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22nd April, 2024

Finite Difference Stencils

"""
import numpy as np 
import torch 

#Order of differentiation, points, order of approximation, dimensionality

def get_stencil(keys):
    dims = keys['dims']
    deriv_order = keys['deriv_orders']
    taylor_order = keys['order']
    points = keys['points']
    

deriv2_3p_2o_1d = torch.tensor([
                                [0, 1, 0],
                                [0, -2, 0],
                                [0, 1, 0]
                                ], dtype=torch.float32)

deriv2_5p_2o_2d = torch.tensor([
                                [0., -1., 0.],
                                [-1., 4., -1.],
                                [0., -1., 0.]
                                ], dtype=torch.float32)

deriv2_9p_4o_2d = torch.tensor([
                                [0, 0, -1/12, 0, 0],
                                [0, 0, 4/3, 0, 0],
                                [-1/12, 4/3, -5/2, 4/3, -1/12],
                                [0, 0, 4/3, 0, 0],
                                [0, 0, -1/12, 0, 0]
                                ], dtype=torch.float32)

deriv2_15p_6o_2d = torch.tensor([
                                [0, 0, 0, 1/90, 0, 0, 0],
                                [0, 0, 0, -3/20, 0, 0, 0],
                                [0, 0, 0, 3/2, 0, 0, 0],
                                [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                                [0, 0, 0, 3/2, 0, 0, 0],
                                [0, 0, 0, -3/20, 0, 0, 0],
                                [0, 0, 0, 1/90, 0, 0, 0]
                                ], dtype=torch.float32)

    
deriv2_2p_2o_1d = torch.tensor([
                                [0, -1, 0],
                                [0, 0, 0],
                                [0, 1, 0]
                                ], dtype=torch.float32)