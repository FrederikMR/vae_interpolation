#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:56:35 2025

@author: fmry
"""

#%% Sources

#%% Modules

#Modules
import torch
import argparse

import timeit

import os

import numpy as np

from load_manifold import load_manifold

#Own files
from torch_geometry.prob_geodesics import ProbGEORCE, ProbGEORCE_Euclidean
from torch_geometry.interpolation import LinearInterpolation, SphericalInterpolation

#%% Code

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--dataset', default='celeba', 
                        type=str)
    parser.add_argument('--data_path', default="~/PhD/Data/CelebA/img", 
                        type=str)
    parser.add_argument('--method', default='Spherical', 
                        type=str)
    parser.add_argument('--lam', default=1.0, 
                        type=float)
    parser.add_argument('--n_points', default=1, 
                        type=int)
    parser.add_argument('--n_sim', default=100, 
                        type=int)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--tol', default=0.001,
                        type=int)
    parser.add_argument('--N', default=10,
                        type=int)
    
    parser.add_argument('--number_repeats', default=1,
                        type=int)
    parser.add_argument('--timing_repeats', default=1,
                        type=int)

    #Continue training or not
    parser.add_argument('--save_path', default='geodesics/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Load manifold

def estimate_time(Geodesic, z0, zT, number_repeats, timing_repeats):
    
    timing = []
    timing = timeit.repeat(lambda: Geodesic(z0,zT), 
                           number=number_repeats, 
                           repeat=timing_repeats)
    timing = np.stack(timing)
    
    return np.mean(timing), np.std(timing)

def compute_interpolation():
    
    #Arguments
    args = parse_args()
    
    x, y, M, model, elbo_fun = load_manifold(args.dataset, args.data_path, args.n_points, args.device)
    
    if not os.path.exists(f'interpolation/{args.dataset}'):
        os.makedirs(f'interpolation/{args.dataset}')
    
    def elbo_evaluate(curve):
        
        elbo = 0.0
        for i in range(args.n_sim):
            elbo += elbo_fun(curve)
        elbo = torch.mean(elbo)
        
        return elbo
    
    def elbo_reg(curve):
        
        return elbo_fun(curve)
    
    if args.method == "ProbGEORCE":
        
        lam_str = str(args.lam).replace('.','d')
        save_name = f"ProbGEORCE_{lam_str}"
        
        Geodesic = ProbGEORCE_Euclidean(reg_fun = elbo_reg,
                                        init_fun=None,
                                        lam=args.lam,
                                        N=args.N,
                                        tol=args.tol,
                                        max_iter=args.max_iter,
                                        line_search_params = {'rho': 0.5},
                                        device=args.device,
                                        )
    elif args.method == "Geodesic":
        
        save_name = "Geodesic"
        
        Geodesic = ProbGEORCE(M=M, 
                              reg_fun = lambda x: torch.sum(torch.zeros_like(x)),
                              init_fun=None,
                              lam=0.0,
                              N=args.N,
                              tol=args.tol,
                              max_iter=args.max_iter,
                              line_search_params = {'rho': 0.5},
                              device=args.device,
                              )
        
    elif args.method == "Linear":
        
        save_name = "Linear"
        
        Geodesic = LinearInterpolation(N=args.N, device=args.device)
    
    elif args.method == "Spherical":
        
        save_name = "Spherical"
        
        Geodesic = SphericalInterpolation(N=args.N, device=args.device)
        
   
    curves = []
    elbos = []
    energy = []
    for x1,y1 in zip(x,y):
        
        hx = model.h(x1).reshape(-1)
        hy = model.h(y1).reshape(-1)

        curve = Geodesic(hx,hy)
        data_curve = model.g(curve)
        curves.append(curve.cpu().detach())
        elbos.append(elbo_evaluate(curve).item())
        
    elbo = -np.mean(np.stack(elbos))
    mean_time, std_time = estimate_time(Geodesic, hx, hy, args.number_repeats, args.timing_repeats)
    
    print(elbo)
    print(mean_time)
    
    torch.save({
            'elbo': elbo,
            'curves': curves,
            'mean_time': mean_time,
            'std_time': std_time,
            }, f'interpolation/{args.dataset}/{save_name}.pt')
    
    return



#%% Calling main

if __name__ == '__main__':
    compute_interpolation()