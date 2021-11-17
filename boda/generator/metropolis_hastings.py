import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

from tqdm import tqdm

class MHBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def collect_samples(self, n_samples=1, n_burnin=0, keep_burnin=False):
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in')
            burnin  = {'samples':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_burnin)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                if keep_burnin:
                    burnin['samples'].append(sample['sample'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_samples >= 1:
            print('collect samples')
            samples = {'samples':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_samples)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                samples['samples'].append(sample['sample'])
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
    
    u = torch.rand_like(old_energy).log()
    accept = u.le( (old_energy-new_energy)/temperature )
    
    sample = torch.stack([old_params, new_params], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    energy = torch.stack([old_energy, new_energy], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    
    params.theta.data = sample # metropolis corrected update
    
    return {'sample': sample.detach().clone().cpu(), 
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
        
class PolynomialDecay(nn.Module):
    def __init__(self,
                 a = 1,
                 b = 1,
                 gamma = 1.,
                ):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def forward(self):
        return self.a*((self.b+self.t)**-self.gamma)
    
    def step(self):
        val = self.forward()
        self.t += 1
        return val
    
    def reset(self):
        self.t = 0
        return None
    
class SimulatedAnnealing(nn.Module):
    def __init__(self, 
                 params,
                 energy_fn, 
                 n_positions=1, 
                 temperature_schedule=PolynomialDecay()
                ):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        self.n_positions = n_positions
        self.temperature_schedule = temperature_schedule
        
        self.mh_engine = naive_mh_step


    def collect_samples(self, n_samples=1, n_burnin=0, keep_burnin=False):
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in')
            burnin  = {'samples':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_burnin)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                if keep_burnin:
                    burnin['samples'].append(sample['sample'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_samples >= 1:
            print('collect samples')
            samples = {'samples':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_samples)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                samples['samples'].append(sample['sample'])
                samples['energies'].append(sample['energy'])
                samples['acceptances'].append(sample['acceptance'])

        samples = { k: torch.stack(v, dim=0) for k,v in samples.items() }
        
        return {'burnin': burnin, 'samples':samples}
