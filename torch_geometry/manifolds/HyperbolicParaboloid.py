#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor

####################

from .manifold import RiemannianManifold

#%% Code

class HyperbolicParaboloid(RiemannianManifold):
    def __init__(self,
                 )->None:

        self.dim = 2
        self.emb_dim = 3
        super().__init__(f=self.f_standard, invf=self.invf_standard)
        
        return
    
    def __str__(self)->str:
        
        return "Hyperbolic Paraboloid equipped with the pull back metric"
    
    def f_standard(self,
                   z:Tensor,
                   )->Tensor:
        
        return torch.hstack((z.T, z[0]**2-z[1]**2))

    def invf_standard(self,
                      x:Tensor,
                      )->Tensor:
        
        return x[:-1]
        