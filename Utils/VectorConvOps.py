#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2nd July, 2024

Vector Operations implemented using the ConvOps Class 
Data used for all operations should be in the shape: BS, Nt, Nx, Ny
"""

from Utils.ConvOps_2d import *

#############################################  
#Vector Operations 
#############################################  

class Divergence(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2):
        super(Divergence, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order)

    def __call__(self, input_x, input_y):

        outputs = self.grad_x(input_x) + self.grad_y(input_y)
        return outputs
    
class Gradient(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2):
        super(Gradient, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order)

    def __call__(self, input_x, input_y):

        outputs = self.grad_x(input_x), self.grad_y(input_y)
        return outputs
    
class Curl(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2):
        super(Curl, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order)

    def __call__(self, input_x, input_y):

        outputs = self.grad_x(input_y) - self.grad_y(input_x)
        return outputs
    

class Laplace(ConvOperator):
    def __init__(self, domain=('x','y'), order=2, scale=1.0, taylor_order=2):
        super(Laplace, self).__init__()

        self.laplace_x = ConvOperator(domain[0], order, scale, taylor_order)
        self.laplace_y = ConvOperator(domain[1], order, scale, taylor_order)

    def __call__(self, input_x, input_y):

        outputs = self.laplace_x(input_x) + self.laplace_y(input_y)
        return outputs
