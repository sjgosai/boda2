import math
import sys
import warnings
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

from tqdm import tqdm

from ..common import utils

class MHBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def collect_samples(self, n_steps=1, n_burnin=0, keep_burnin=False):
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in', file=sys.stderr)
            burnin  = {'states':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_burnin)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                if keep_burnin:
                    burnin['states'].append(sample['state'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_steps >= 1:
            print('collect samples', file=sys.stderr)
            samples = {'samples':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_steps)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                samples['states'].append(sample['state'])
                samples['energies'].append(sample['energy'])
                samples['acceptances'].append(sample['acceptance'])

        samples = { k: torch.stack(v, dim=0) for k,v in samples.items() }
        
        return {'burnin': burnin, 'samples':samples}

@torch.no_grad()
def naive_mh_step(params, energy_fn, n_positions=1, temperature=1.0):
    
    assert len(params.shape) == 3
    
    old_params = params.theta.detach().clone()
    old_seq    = params()
    old_energy = energy_fn(old_seq)
    old_energy = params.rebatch( old_energy )
    old_nll = old_params * -115
    
    pos_shuffle = torch.argsort( torch.rand(old_params.shape[0], old_params.shape[-1]), dim=-1 )
    proposed_positions = pos_shuffle[:,:n_positions]
    batch_slicer = torch.arange(old_params.shape[0]).view(-1,1)
    
    updates = old_nll[batch_slicer, :, proposed_positions].mul(-1)
    old_nll[batch_slicer, :, proposed_positions] = updates
    
    proposal_dist = dist.OneHotCategorical(logits=-old_nll.permute(0,2,1)/temperature)
    
    new_params = proposal_dist.sample().permute(0,2,1).detach().clone()
    params.theta.data = new_params # temporary update
    new_seq    = params()
    new_energy = energy_fn(new_seq)
    new_energy = params.rebatch( new_energy )
    
    u = torch.rand_like(old_energy).log()
    accept = u.le( (old_energy-new_energy)/temperature )
    
    sample = torch.stack([old_params, new_params], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    energy = torch.stack([old_energy, new_energy], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    
    params.theta.data = sample # metropolis corrected update
    
    return {'state': sample.detach().clone().cpu(), 
            'energy': energy.detach().clone().cpu(), 
            'acceptance': accept.detach().clone().cpu()}

class NaiveMH(MHBase):
    def __init__(self, 
                 energy_fn, 
                 params,
                 n_positions=1, 
                 temperature=1.0
                ):
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params
        self.n_positions = n_positions
        self.temperature = temperature
        
        self.mh_kwargs = {'n_positions': self.n_positions, 
                          'temperature': self.temperature}
        
        self.mh_engine = naive_mh_step
        
class PolynomialDecay:
    def __init__(self,
                 a = 1,
                 b = 1,
                 gamma = 1.,
                ):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def __call__(self):
        return self.a*((self.b+self.t)**-self.gamma)
    
    def step(self):
        val = self()
        self.t += 1
        return val
    
    def reset(self):
        self.t = 0
        return None

class SimulatedAnnealing(nn.Module):
    
    @staticmethod
    def add_generator_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        group  = parser.add_argument_group('Generator Constructor args')
        group.add_argument('--n_positions', type=int, default=1)
        group.add_argument('--a', type=float, default=1.)
        group.add_argument('--b', type=float, default=1.)
        group.add_argument('--gamma', type=float, default=1.)
        
        group  = parser.add_argument_group('Generator Runtime args')
        group.add_argument('--energy_threshold', type=float, default=float("Inf"))
        group.add_argument('--max_attempts', type=int, default=10000)
        group.add_argument('--n_steps', type=int, default=1)
        group.add_argument('--n_burnin', type=int, default=0)
        group.add_argument('--keep_burnin', type=utils.str2bool, default=False)
        return parser

    @staticmethod
    def process_args(grouped_args):
        constructor_args = grouped_args['Generator Constructor args']
        runtime_args     = grouped_args['Generator Runtime args']
        
        return constructor_args, runtime_args
    
    def __init__(self, 
                 params,
                 energy_fn, 
                 n_positions=1, 
                 a=1.,
                 b=1.,
                 gamma=1.,
                ):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        self.n_positions = n_positions
        self.a = a
        self.b = b
        self.gamma = gamma
        self.temperature_schedule = PolynomialDecay(a,b,gamma)
        
        self.mh_engine = naive_mh_step

    def collect_samples(self, n_steps=1, n_burnin=0, keep_burnin=False):
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in', file=sys.stderr)
            burnin  = {'states':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_burnin)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                if keep_burnin:
                    burnin['states'].append(sample['state'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_steps >= 1:
            print('collect samples', file=sys.stderr)
            samples = {'states':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_steps)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                samples['states'].append(sample['state'])
                samples['energies'].append(sample['energy'])
                samples['acceptances'].append(sample['acceptance'])

        samples = { k: torch.stack(v, dim=0) for k,v in samples.items() }
        
        return {'burnin': burnin, 'samples':samples}

    def generate(self, n_proposals=1, energy_threshold=float("Inf"), 
                 max_attempts=10000, n_steps=1, n_burnin=0, keep_burnin=False):
        
        batch_size, *theta_shape = self.params.theta.shape
        proposals = torch.randn([0,*theta_shape])
        energies  = torch.randn([0])
        
        attempts = 0
        FLAG = True
        
        while (proposals.shape[0] < n_proposals) and FLAG:
            
            attempts += 1
            FLAG = attempts < max_attempts
            
            trajectory = self.collect_samples(
                n_steps=n_steps, n_burnin=n_burnin, keep_burnin=keep_burnin
            )
            
            final_states  = trajectory['samples']['states'][-1]
            self.params.theta.data = final_states
            final_energies = self.energy_fn.energy_calc( self.params() )
            final_energies = self.params.rebatch( final_energies ) \
                               .detach().clone().cpu()
            
            energy_filter = final_energies <= energy_threshold
            
            proposals = torch.cat([proposals,  final_states[energy_filter]], dim=0)
            energies  = torch.cat([energies, final_energies[energy_filter]], dim=0)
            
        return {'proposals': proposals[:n_proposals], 'energies': energies[:n_proposals]}


