import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from torch.distributions.categorical import Categorical

class BasicParameters(nn.Module):
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
    
class StraightThroughParameters(nn.Module):
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
        self.batch_size = self.theta.shape[0]
        
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