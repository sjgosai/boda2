import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

from boda.generator.plot_tools import ppm_to_IC, ppm_to_pwm
from boda import common

boda_src = os.path.join( os.path.dirname( os.path.dirname( os.getcwd() ) ), 'src' )
sys.path.insert(0, boda_src)
from pymeme import streme, parse_streme_output

class BaseEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = None
        
    def forward(self, x_in):

        hook = self.energy_calc(x_in)
        
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
    def __init__(self, model, bias_cell=0, bias_alpha=1., score_pct=.3):
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        self.score_pct = score_pct
        
    def energy_calc(self, x):
        hook = x.to(self.model.device)
        
        hook = self.model(hook)
        
        return hook[...,[ x for x in range(hook.shape[-1]) if x != self.bias_cell]].max(-1).values \
                 - hook[...,self.bias_cell].mul(self.bias_alpha)
    
    def penalty(self, x):
        return self.motif_penalty(x)

    def register_penalty(self, x):
        try:
            self.penalty_filters = x.type_as(self.penalty_filters)
        except AttributeError:
            self.register_buffer('penalty_filters', x)
            
    def register_threshold(self, x):
        try:
            self.score_thresholds = x.type_as(self.score_thresholds)
        except AttributeError:
            self.register_buffer('score_thresholds', x)
            
    def streme_penalty(self, streme_output):
        
        try:
            penalty_weight = (self.penalty_filters.shape[0] // 2) + 1
        except AttributeError:
            penalty_weight = 1
        
        motif_data = parse_streme_output(streme_output['output'])
        top_ppm    = common.utils.align_to_alphabet( 
            motif_data['motif_results'][0]['ppm'], 
            motif_data['meta_data']['alphabet'], 
            common.constants.STANDARD_NT 
        )
        top_ppm = torch.tensor(top_ppm).float()
        background = [ motif_data['meta_data']['frequencies'][nt] 
                       for nt in common.constants.STANDARD_NT ]
        top_pwm = ppm_to_pwm(top_ppm, background) * (penalty_weight**0.33) # (4, L)
        max_score = torch.max(top_pwm, dim=0)[0].sum()
        top_pwm_rc = common.utils.reverse_complement_onehot(top_pwm) # (4, L)

        proposed_penalty = torch.stack([top_pwm, top_pwm_rc] ,dim=0) # (2, 4, L)
        proposed_thresholds = torch.tensor(2 * [self.score_pct * max_score]) # (2,)
        
        try:
            penalty_filters = torch.cat(
                [self.penalty_filters, proposed_penalty.to(self.penalty_filters.device)], 
                dim=0
            ) # (2k+2, 4, L)
            score_thresholds= torch.cat(
                [self.score_thresholds, proposed_thresholds.to(self.score_thresholds.device)]
            ) # (2k+2,)
            
        except AttributeError:
            penalty_filters = proposed_penalty.to(self.model.device)
            score_thresholds= proposed_thresholds.to(self.model.device)
            
        self.register_penalty(penalty_filters)
        self.register_threshold(score_thresholds)
        
        
    def motif_penalty(self, x):
        try:
            motif_scores = F.conv1d(x, self.penalty_filters)
            print('scores shape: {}'.format(motif_scores.shape))
            score_thresholds = torch.ones_like(motif_scores) * self.score_thresholds[None, :, None]
            print('thresh shape: {}'.format(score_thresholds.shape))
            mask = torch.ge(motif_scores, score_thresholds)
            print('mask shape: {}'.format(mask.shape))
            #masked_scores = torch.masked_select(motif_scores, mask)
            masked_scores = motif_scores * mask.float()
            print('masked scores shape: {}'.format(masked_scores.shape))
            return masked_scores.flatten(1).sum(dim=-1).div((self.penalty_filters.shape[0] // 2) * x.shape[0])

        except AttributeError:
            return 0
    
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
    
