import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from torch.distributions.categorical import Categorical

class ParamsBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    @property
    def shape(self):
        return self().shape
    
    def forward(self):
        raise NotImplementedError("Params forward call not implemented.")
        return None
        
    def rebatch(self, input):
        raise NotImplementedError("Rebatch function not implemented.")
        return None
    
    def prior_nll(self):
        raise NotImplementedError("Prior Negative Log-Likelihood not implemented.")
        return None
    
class BasicParameters(ParamsBase):
    def __init__(self,
                 data,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1
                ):
        
        super().__init__()
        
        self.register_parameter('theta', data)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        
    @property
    def shape(self):
        return self().shape

    def forward(self):
        my_attr = [ getattr(self, x) for x in ['theta', 'left_flank', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def rebatch(self, input):
        return input

class StraightThroughParameters(ParamsBase):
    def __init__(self,
                 data,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1,
                 n_samples=1,
                 temperature=1.,
                ):
        
        super().__init__()
        
        self.register_parameter('theta', data)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.n_samples = n_samples
        self.temperature = temperature
        
        self.softmax    = nn.Softmax(dim=-1)
        self.num_classes= self.theta.shape[1]
        self.n_dims     = len(self.theta.shape)
        self.batch_size = self.theta.shape[batch_dim]
        
        self.perm_order   = [0] + list(range(2,self.n_dims)) + [1]
        self.reverse_perm = [0,self.n_dims-1] + list(range(1,self.n_dims-1))
                
    @property
    def shape(self):
        return get_logits().shape
    
    def get_logits(self):
        my_attr = [ getattr(self, x) for x in ['theta', 'left_flank', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def get_probs_and_dist(self):
        logits = self.get_logits()
        logits = logits.permute( *self.perm_order ) \
                   .div(self.temperature)
        probs  = self.softmax(logits)
        dist   = Categorical(probs=probs)
        return probs, dist
    
    def sample(self):
        probs, dist = self.get_probs_and_dist()
        sample = dist.sample((self.n_samples,))
        sample = F.one_hot(sample, self.num_classes)
        sample = sample - probs.detach() + probs
        sample = sample.flatten(0,1)
        sample = sample.permute( *self.reverse_perm )
        return sample
    
    def forward(self):
        return self.sample()
    
    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)
    
class GumbelSoftmaxParameters(ParamsBase):
    def __init__(self,
                 data, 
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1,
                 n_samples=1,
                 tau=1.,
                 prior_var=1.
                ):
        
        super().__init__()
        self.register_parameter('theta', data)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.n_samples = n_samples
        self.tau = tau
        self.prior_var = prior_var
        
        self.softmax    = nn.Softmax(dim=-1)
        self.num_classes= self.theta.shape[1]
        self.n_dims     = len(self.theta.shape)
        self.batch_size = self.theta.shape[batch_dim]
        
    @property
    def shape(self):
        return self.get_logits().shape
    
    def get_logits(self):
        my_attr = [ getattr(self, x) for x in ['theta', 'left_flank', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def get_sample(self):
        logits = self.get_logits()
        samples= [ F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1) 
                   for i in range(self.n_samples) ]
        return torch.stack(samples, dim=0)
    
    def forward(self):
        sample = self.get_sample()
        return sample.flatten(0,1)
        
    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)
    
#     def prior_nll(self):
#         return self.theta.transpose(self.batch_dim, 0).flatten(1).pow(2).div(2*self.prior_var).mean(1)