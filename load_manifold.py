# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:44:58 2021

@author: Frederik
"""

#%% Modules

#Modules
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pandas as pd

from VAE_celeba import VAE_CELEBA
from VAE_svhn import VAE_SVHN
from VAE_surface3d import VAE_3d

#Own files
from torch_geometry.manifolds import LatentSpaceManifold

#%% Load manifold

def load_manifold(dataset:str,
                  data_path:str,
                  n_points:int=10,
                  device:str='cpu',
                  ):
    
    if dataset == "celeba":
        
        dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
        #Plotting the trained model
        model = VAE_CELEBA().to(device) #Model used
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        
        checkpoint = torch.load('trained_models/celeba/celeba.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.eval()
        
        #Get 3 images to compute geodesics for
        subset_indices = list(range(n_points*2)) # select your indices here as a list
        dataset_subset = torch.utils.data.Subset(dataset, subset_indices)
        
        x, y = [], []
        for i in range(n_points):
            x.append((dataset_subset[2*i][0]).view(1, 3, 64, 64))
            y.append((dataset_subset[2*i+1][0]).view(1, 3, 64, 64))
            
        x, y = torch.stack(x), torch.stack(y)
        
        M = LatentSpaceManifold(32, 
                                64*64*3, 
                                model.h, 
                                lambda x: model.g(x.reshape(1,-1)).reshape(-1),
                                )
        
        return x, y, M, model, model.elbo
    
    elif dataset == "svhn":
        
        dataset = dset.SVHN(root=data_path, split = 'train',
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
        
        #Plotting the trained model
        model = VAE_SVHN().to(device) #Model used
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        
        checkpoint = torch.load('trained_models/svhn/svhn.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.eval()
        
        #Get 3 images to compute geodesics for
        subset_indices = list(range(n_points*2)) # select your indices here as a list
        dataset_subset = torch.utils.data.Subset(dataset, subset_indices)
        
        x, y = [], []
        for i in range(n_points):
            x.append((dataset_subset[2*i][0]).view(1, 3, 32, 32))
            y.append((dataset_subset[2*i+1][0]).view(1, 3, 32, 32))
            
        x, y = torch.stack(x), torch.stack(y)
        
        M = LatentSpaceManifold(32, 
                                32*32*3, 
                                model.h, 
                                lambda x: model.g(x.reshape(1,-1)).reshape(-1),
                                )
        
        return x, y, M, model, model.elbo
    
    else:
        
        df = pd.read_csv(data_path, index_col=0)
        DATA = torch.Tensor(df.values, device=device)
        DATA = torch.transpose(DATA, 0, 1)
        
        #Loading model
        model = VAE_3d().to(device) #Model used
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        checkpoint = torch.load(f'trained_models/{dataset}/{dataset}.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.eval()
        
        x = DATA[:n_points]
        y = DATA[n_points:(2*n_points)]
        
        M = LatentSpaceManifold(2, 
                                3, 
                                model.h, 
                                lambda x: model.g(x.reshape(1,-1)).reshape(-1),
                                )
        
        return x, y, M, model, model.elbo
    
    return 