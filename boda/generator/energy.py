import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

boda_src = os.path.join( os.path.dirname( os.path.dirname( os.getcwd() ) ), 'src' )
sys.path.insert(0, boda_src)
from pymeme import streme, parse_streme_output

class BaseEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = None
        
    def forward(self, x):

        hook = self.energy_calc(hook)
        
        try:
            pen = self.penalty(x_in)
            hook = hook + pen
        except NotImplementedError:
            pass
        
        return hook
      
    def energy_calc(self, x):
        raise NotImplementedError("Energy caclulation not implemented.")
        x_in = x.to(self.model.device)
        
        hook = self.model(x_in)
        # do math

        return hook
      
    def penalty(self, x):
        raise NotImplementedError("Penalty not implemented")
        
        hook = x
        
        return hook

class OverMaxEnergy(BaseEnergy):
    def __init__(self, model, bias_cell=0, bias_alpha=1.):
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        
    def forward(self, x):
        hook = x.to(self.model.device)
        
        hook = self.model(hook)
        
        return hook[...,[ x for x in range(hook.shape[-1]) if x != self.bias_cell]].max(-1).values \
                 - hook[...,self.bias_cell].mul(self.bias_alpha)
    
    def register_penalty(self, x):
        try:
            self.penalty = x.type_as(self.penalty)
        except AttributeError:
            self.register_buffer('penalty', x)
            
    def streme_penalty(self, streme_output):
        motif_data = parse_streme_output(streme_results['output'])
        top_ppm    = motif_data['motif_results'][0]['ppm']
        
    
class EntropyEnergy(BaseEnergy):
    def __init__(self, model, bias_cell=None, bias_alpha=1.):
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        
    def forward(self, x):
        hook   = x.to(self.model.device)
        
        hook   = self.model(hook)

        energy = boda.graph.utils.shannon_entropy(hook)
        
        if self.bias_cell is not None:
            energy = energy - hook[...,self.bias_cell].mul(self.bias_alpha)
        
        return energy
    
