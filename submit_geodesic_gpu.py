#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_geodesic.sh")
    
    return

#%% Generate jobs

def generate_job(dataset, data_path, method, lam):

    with open ('submit_geodesic.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J {dataset}_{method}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 geodesic.py \\
        --dataset {dataset} \\
        --data_path {data_path} \\
        --method {method} \\
        --lam {lam} \\
        --n_points 10 \\
        --device gpu \\
        --max_iter 1000 \\
        --tol 0.0001 \\
        --N 10 \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --save_path geodesics_gpu/
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    models = {'svhn': '/work3/fmry/Data/SVHN/',
              'celeba': '/work3/fmry/Data/CelebA/'}
    methods = ['Linear', 'Spherical', 'Geodesic']
    pgeorce = {'ProbGEORCE': [0.1, 0.5, 1.0]}
    
    for dataset, datapath in models.items():
        for lam in pgeorce.values():
            generate_job(dataset, datapath, "ProbGEORCE", lam)
            try:
                submit_job()
            except:
                time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                try:
                    submit_job()
                except:
                    print(f"Job script for {dataset} with method ProbGEORCE failed!")
        for m in methods:
            generate_job(dataset, datapath, m, 0.0)
            try:
                submit_job()
            except:
                time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                try:
                    submit_job()
                except:
                    print(f"Job script for {dataset} with method {m} failed!")
    return

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)
