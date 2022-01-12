import os
import sys
import argparse
import math
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

from boda.generator.plot_tools import ppm_to_IC, ppm_to_pwm
from boda import common

from boda.common.utils import unpack_artifact, model_fn
from boda.common.pymeme import streme, parse_streme_output

######################
##                  ##
## Energy Functions ##
##                  ##
######################

class BaseEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = None
        
    def forward(self, x_in):

        hook = self.energy_calc(x_in)
        
        try:
            pen = self.penalty(x_in)
            hook = hook + pen
        except AttributeError:
            try:
                _ = self.you_have_been_warned
            except AttributeError:
                print("Penalty not implemented", file=sys.stderr)
                self.you_have_been_warned = True
            pass
        
        return hook
      
    def energy_calc(self, x):
        raise NotImplementedError("Energy caclulation not implemented.")
        x_in = x.to(self.model.device)
        
        hook = self.model(x_in)
        # do math

        return hook
      
class OverMaxEnergy(BaseEnergy):
    
    @staticmethod
    def add_energy_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Energy Module args')
        group.add_argument('--model_artifact', type=str)
        group.add_argument('--bias_cell', type=int, default=0)
        group.add_argument('--bias_alpha', type=float, default=1.)
        group.add_argument('--bending_factor', type=float, default=0.)
        group.add_argument('--a_min', type=float, default=-math.inf)
        group.add_argument('--a_max', type=float, default=math.inf)
        return parser

    @staticmethod
    def process_args(grouped_args):
        energy_args = grouped_args['Energy Module args']
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            unpack_artifact(energy_args.model_artifact, tmpdirname)
            model = model_fn(os.path.join(tmpdirname,'artifacts'))
        model.cuda()
        model.eval()
        energy_args.model = model

        del energy_args.model_artifact
        
        return energy_args

    def __init__(self, model, bias_cell=0, bias_alpha=1., bending_factor=0., a_min=-math.inf, a_max=math.inf):
        super().__init__()
        
        self.model = model
        self.model.eval()

        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        self.bending_factor = bending_factor
        self.a_min = a_min
        self.a_max = a_max

    def bend(self, x):
        return x - self.bending_factor * (torch.exp(-x) - 1)
        
    def energy_calc(self, x):
        hook = x.to(self.model.device)
        
        hook = self.bend(self.model(hook).clamp(self.a_min,self.a_max))
        energy = hook[...,[ x for x in range(hook.shape[-1]) if x != self.bias_cell]].max(-1).values \
                 - hook[...,self.bias_cell].mul(self.bias_alpha)
        return energy

class TargetEnergy(BaseEnergy):
    
    @staticmethod
    def add_energy_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Energy Module args')
        group.add_argument('--model_artifact', type=str)
        group.add_argument('--targets', type=float, nargs='+')
        group.add_argument('--lambd', type=float, default=0.0)
        group.add_argument('--a_min', type=float, default=-math.inf)
        group.add_argument('--a_max', type=float, default=math.inf)
        return parser

    @staticmethod
    def process_args(grouped_args):
        energy_args = grouped_args['Energy Module args']
        
        unpack_artifact(energy_args.model_artifact)
        model = model_fn('./artifacts')
        model.cuda()
        model.eval()
        energy_args.model = model

        del energy_args.model_artifact
        
        return energy_args

    def __init__(self, model, targets, lambd=0.0, a_min=-math.inf, a_max=math.inf):
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.lambd = lambd
        self.register_buffer(
            'targets',torch.tensor(targets).to(self.model.device)
        )
        
        self.shrink = nn.Softshrink(lambd=self.lambd)
        
        self.a_min = a_min
        self.a_max = a_max

    def energy_calc(self, x):
        hook = x.to(self.model.device)
        
        energy = self.shrink( self.model(hook).clamp(self.a_min, self.a_max) - self.targets ).abs().sum(-1)
        
        return energy

        
class EntropyEnergy(BaseEnergy):
    def __init__(self, model, bias_cell=None, bias_alpha=1.):
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        
    def energy_calc(self, x):
        hook   = x.to(self.model.device)
        
        hook   = self.model(hook)

        energy = boda.graph.utils.shannon_entropy(hook)
        
        if self.bias_cell is not None:
            energy = energy - hook[...,self.bias_cell].mul(self.bias_alpha)
        
        return energy
    

####################
##                ##
## Penalty MixIns ##
##                ##
####################

class BasePenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def penalty(self, x):
        raise NotImplementedError("Penalty not implemented")
        
        hook = x
        
        return hook

def sync_width(tensor_1, tensor_2):
    
    bs_1, nc_1, ln_1 = tensor_1.shape
    bs_2, nc_2, ln_2 = tensor_2.shape
    
    if ln_1 != ln_2:
        if ln_1 < ln_2:
            ln_d = ln_2 - ln_1
            padded_1 = F.pad(tensor_1,(0,ln_d),mode='constant',value=0.)
            return padded_1, tensor_2
        else:
            ln_d = ln_1 - ln_2
            padded_2 = F.pad(tensor_2,(0,ln_d),mode='constant',value=0.)
            return tensor_1, padded_2
    else:
        return tensor_1, tensor_2

    
class StremePenalty(BasePenalty):
    @staticmethod
    def add_penalty_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Penalty Module args')
        group.add_argument('--score_pct', type=float, default=0.3)
        return parser

    @staticmethod
    def process_args(grouped_args):
        penalty_args = grouped_args['Penalty Module args']
        return penalty_args
    
    def __init__(self, score_pct):
        super().__init__()
        
        self.score_pct = score_pct

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
        #background = [ motif_data['meta_data']['frequencies'][nt] 
        #               for nt in common.constants.STANDARD_NT ]
        background = 4*[0.25]
        top_pwm = ppm_to_pwm(top_ppm, background) * (penalty_weight**0.33) # (4, L)
        max_score = torch.max(top_pwm, dim=0)[0].sum()
        top_pwm_rc = common.utils.reverse_complement_onehot(top_pwm) # (4, L)

        proposed_penalty = torch.stack([top_pwm, top_pwm_rc] ,dim=0) # (2, 4, L)
        proposed_thresholds = torch.tensor(2 * [self.score_pct * max_score]) # (2,)
        
        try:
            penalty_filters = torch.cat(
                sync_width(self.penalty_filters, proposed_penalty.to(self.penalty_filters.device)), 
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
            score_thresholds = torch.ones_like(motif_scores) * self.score_thresholds[None, :, None]
            mask = torch.ge(motif_scores, score_thresholds)
            #masked_scores = motif_scores * mask.float()
            #return masked_scores.flatten(1).sum(dim=-1).div((self.penalty_filters.shape[0] // 2) * x.shape[0])
        
            masked_scores = torch.masked_select(motif_scores, mask)
            return masked_scores.sum(dim=-1).mean().div((self.penalty_filters.shape[0] // 2) * x.shape[0])

        except AttributeError:
            return 0

    def penalty(self, x):
        hook = x.to(self.model.device)
        return self.motif_penalty(hook)

    def update_penalty(self, proposal):
        proposals_list = common.utils.batch2list(proposal['proposals'])
        streme_results = streme(proposals_list)
        self.streme_penalty(streme_results)
        update_summary = {
            'streme_output': streme_results,
            'filters': self.penalty_filters.detach().clone(),
            'score_thresholds': self.score_thresholds.detach().clone()
        }
        return update_summary
