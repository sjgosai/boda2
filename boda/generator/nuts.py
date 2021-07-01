import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.categorical import Categorical

from tqdm import tqdm

class LeapfrogBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def calc_energy(self, T=1.0):
        energy = self.energy_fn(self.params()).div(T)
        energy = self.params.rebatch( energy )
        try:
            prior = self.params.prior_nll()
            energy= energy + prior
        except NotImplementedError:
            warnings.warn("Prior Negative Log-Likelihood Not Implemented.", RuntimeWarning)
            pass
        return energy
    
    def leapfrog(self, theta, r, epsilon, T=1.0):
        
        #print(f'leap in params:\n{theta[0,:,100]}')
        #print(f'leap in momentum:\n{r[0,:,100]}')
        self.params.theta.data = theta
        self.params.zero_grad()
        energy = self.calc_energy(T)
        grad_U = ag.grad( energy.sum(), self.params.theta )[0]
        #print(f'first grad:\n{grad_U[0,:,100]}')
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            #print(f'first momentum update:\n{r[0,:,100]}')
            
            theta = theta + r.mul(epsilon)
            #print(f'params update:\n{theta[0,:,100]}')
            
        self.params.theta.data = theta
        self.params.zero_grad()
        energy = self.calc_energy(T)
        grad_U = ag.grad( energy.sum(), self.params.theta )[0]
        #print(f'second grad:\n{grad_U[0,:,100]}')
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            #print(f'second momentum update:\n{r[0,:,100]}')
            
        #print(f'leap out:\n{theta[0,:,100]}')
        #print('')
        return theta, r, energy, grad_U
    
    def eval_samples(self, sample_tensor):
        with torch.no_grad():
            nlls = []
            for theta in sample_tensor:
                self.params.theta.data = theta
                self.params.zero_grad()
                nll = self.energy_fn(self.params())
                nll = self.params.rebatch( nll )
                nlls.append( nll.clone().detach().cpu() )
        return torch.stack(nlls, dim=0)

class TemperatureScheduleBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def step(inner=None, outer=None):
        return None
        
    def forward():
        return 1.0
    
class WaveSchedule(TemperatureScheduleBase):
    def __init__(self, T_init=1.0, T_max=None, T_min=None, 
                 outer_warmup=None, outer_cooldown=None, 
                 inner_warmup=None, inner_cooldown=None):
        super().__init__()
        
        self.T_init = T_init
        self.T_max  = T_max
        self.T_min  = T_min
        
        self.outer_warmup = outer_warmup
        self.outer_cooldown = outer_cooldown
        self.inner_warmup = inner_warmup
        self.inner_cooldown = inner_cooldown
        
        self.use_outer = (outer_warmup is not None) and (outer_cooldown is not None)
        self.use_inner = (inner_warmup is not None) and (inner_cooldown is not None)
        
        self.outer_step = 0
        self.inner_step = 0
        
        self.T_hist = []
        
    def step(self, inner=None, outer=None):
        
        if inner is not None:
            self.inner_step = inner
            
        if outer is not None:
            self.outer_step = outer
            
        return {'outer_step':self.outer_step, 
                'inner_step':self.inner_step}
    
    def get_inner_T(self):
        if self.inner_step < self.inner_warmup:
            hook = self.inner_step/self.inner_warmup
            hook = (self.T_max - self.T_init)*hook
            hook = hook + self.T_init
        elif self.inner_step < self.inner_warmup+self.inner_cooldown:
            hook = (self.inner_step-(self.inner_warmup+self.inner_cooldown)) / self.inner_cooldown
            hook = -((1 - hook**2)**0.5)
            hook = hook*(self.T_max)
            hook = hook + self.T_max + self.T_min
        else:
            hook = 0.
            
        return hook
    
    def get_outer_T(self):
        if self.outer_step < self.outer_warmup:
            hook = self.T_init + (self.outer_step/self.outer_warmup)
        elif self.outer_step < self.outer_warmup+self.outer_cooldown:
            hook = (1+self.T_init)*(1 - ((self.outer_step-self.outer_warmup)/self.outer_cooldown))
        else:
            hook = 0.
        return hook
    
    def forward(self):
        if self.use_inner and not self.use_outer:
            hook = self.get_inner_T()
        elif self.use_outer and not self.use_inner:
            hook = self.get_outer_T() - self.T_init
            hook = hook * (self.T_max/2)
            hook = hook + self.T_init
        elif self.use_outer and self.use_inner:
            hook = self.get_outer_T() * self.get_inner_T()
        else:
            hook = self.T_init
            
        hook = max(self.T_min, min(self.T_max, hook))
        self.T_hist.append(hook)
            
        return hook

class GDTest(LeapfrogBase):
    def __init__(self,
                 params,
                 energy_fn
                ):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
    def collect_samples(self, epsilon, n_samples=1):
        
        samples = []
        theta_m = self.params.theta.clone().detach()
        
        for m in range(n_samples):
            r_0 = torch.zeros_like(theta_m)
            theta_m, r_m, U_m, grad_U = self.leapfrog(theta_m, r_0, epsilon)
            samples.append( 
                {'params':theta_m.clone().detach().cpu(), 
                 'energy': U_m.clone().detach().cpu(), 
                 'grad': grad_U.clone().detach().cpu()} 
            )
                
        return {'samples': samples}
    
class HMC(LeapfrogBase):
    def __init__(self,
                 params,
                 energy_fn
                ):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
        
    def sample_trajectory(self, theta, epsilon, L, inertia=1., alpha=1.):
        theta_0 = theta
        self.params.theta.data = theta
        r = torch.randn_like( theta ).div(inertia)
        
        with torch.no_grad():
            c_U = self.calc_energy()
            c_K = r.pow(2).flatten(1).sum(1).div(2.)
        
        for i in range(L//2):
            r = r.mul(alpha**0.5)
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon)
            r = r.mul(alpha**0.5)
            
        if L % 2 == 1:
            r = r.mul(alpha**0.5)
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon)
            r = r.div(alpha**0.5)
            
        for i in range(L//2):
            r = r.div(alpha**0.5)
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon)
            r = r.div(alpha**0.5)
            
        r = r.mul(-1)
        
        with torch.no_grad():
            
            p_U = self.calc_energy()
            p_K = r.pow(2).flatten(1).sum(1).div(2.)
            
            accept = torch.rand_like(c_U) < (c_U - p_U + c_K - p_K).exp()
            
            theta_p = torch.stack([theta_0, theta], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            U       = torch.stack([c_U, p_U], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            
            return theta_p, U, accept

    def collect_samples(self, epsilon, L, inertia=1., alpha=1., n_samples=1, n_burnin=0):
        burnin_history = min(n_burnin // 10, 50)
        
        samples = []
        burnin  = []
        theta_m = self.params.theta.clone().detach()
        
        for m in tqdm(range(n_burnin)):
            theta_m, U_m, accept_m = self.sample_trajectory( theta_m, epsilon, L, inertia, alpha )
            with torch.no_grad():
                burnin.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach()} 
                )
            if len(burnin[-burnin_history:]) == burnin_history and m % burnin_history == 0:
                aap = torch.stack([ x['acceptance'] for x in burnin[-burnin_history:] ], dim=0) \
                        .float().mean(dim=0)
                try:
                    epsilon[ aap < 0.55 ] *= 0.8
                    epsilon[ aap > 0.70 ] *= 1.25
                except (IndexError, TypeError) as e:
                    if aap.mean() < 0.55:
                        epsilon *= 0.8
                    elif aap.mean() > 0.70:
                        epsilon *= 1.25

        for m in tqdm(range(n_samples)):
            theta_m, U_m, accept_m = self.sample_trajectory( theta_m, epsilon, L, inertia, alpha )
            with torch.no_grad():
                samples.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach()} 
                )
            
        return {'samples': samples, 'burnin': burnin}

class AnnealingHMC(LeapfrogBase):
    def __init__(self,
                 params,
                 energy_fn,
                 temperature_schedule
                ):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        self.temperature_schedule = temperature_schedule
        
    def sample_trajectory(self, theta, epsilon, L, inertia=1., alpha=1.):
        theta_0 = theta
        self.params.theta.data = theta
        r = torch.randn_like( theta ).div(inertia)
        
        with torch.no_grad():
            c_U = self.calc_energy()
            c_K = r.pow(2).flatten(1).sum(1).div(2.)
        
        for i in range(L//2):
            r = r.mul(alpha**0.5)
            T = self.temperature_schedule()
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon, T)
            self.temperature_schedule.step(inner=i)
            r = r.mul(alpha**0.5)
            
        if L % 2 == 1:
            i += 1
            r = r.mul(alpha**0.5)
            T = self.temperature_schedule()
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon, T)
            self.temperature_schedule.step(inner=i)
            r = r.div(alpha**0.5)
            
        for j in range(L//2):
            r = r.div(alpha**0.5)
            T = self.temperature_schedule()
            theta, r, energy, grad_U = self.leapfrog(theta, r, epsilon, T)
            self.temperature_schedule.step(inner=i+j)
            r = r.div(alpha**0.5)
            
        r = r.mul(-1)
        
        with torch.no_grad():
            
            p_U = self.calc_energy()
            p_K = r.pow(2).flatten(1).sum(1).div(2.)
            
            accept = torch.rand_like(c_U) < (c_U - p_U + c_K - p_K).exp()
            
            theta_p = torch.stack([theta_0, theta], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            U       = torch.stack([c_U, p_U], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            
            return theta_p, U, accept

    def collect_samples(self, epsilon, L, inertia=1., alpha=1., n_samples=1, n_burnin=0):
        burnin_history = min(n_burnin // 10, 50)
        
        samples = []
        burnin  = []
        theta_m = self.params.theta.clone().detach()
        
        for m in range(n_burnin):
            theta_m, U_m, accept_m = self.sample_trajectory( theta_m, epsilon, L, inertia, alpha )
            self.temperature_schedule.step(outer=m)
            with torch.no_grad():
                burnin.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach()} 
                )
            if len(burnin[-burnin_history:]) == burnin_history and m % burnin_history == 0:
                aap = torch.stack([ x['acceptance'] for x in burnin[-burnin_history:] ], dim=0) \
                        .float().mean(dim=0)
                try:
                    epsilon[ aap < 0.55 ] *= 0.8
                    epsilon[ aap > 0.70 ] *= 1.25
                except (IndexError, TypeError) as e:
                    if aap.mean() < 0.55:
                        epsilon *= 0.8
                    elif aap.mean() > 0.70:
                        epsilon *= 1.25

        for m in range(n_samples):
            theta_m, U_m, accept_m = self.sample_trajectory( theta_m, epsilon, L, inertia, alpha )
            self.temperature_schedule.step(outer=m+n_burnin)
            with torch.no_grad():
                samples.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach()} 
                )
            
        return {'samples': samples, 'burnin': burnin}

class NUTS3(LeapfrogBase):
    def __init__(self,
                 params,
                 energy_fn,
                 max_tree_depth=10
                ):
        
        super().__init__()
        self.params = params
        self.energy_fn  = energy_fn
        self.max_tree_depth = max_tree_depth
        
        self.d_max = 1000.
        
    def buildtree(self, theta, r, u, v, j, epsilon):
        #print(f'current j: {j}')
        if j == 0:
            theta_p, r_p, energy_p, grad_p = self.leapfrog(theta, r, v*epsilon)
            batch_dot = torch.einsum('bs,bs->b', r_p.flatten(1), r_p.flatten(1))
            hamilton  = energy_p + batch_dot.div(2.)
            n_p = (u <= torch.exp(-hamilton)).type(torch.long)
            s_p = (torch.log(u).add(-self.d_max) < -hamilton).type(torch.long)
            #print(f'inner j: {j}')
            #print(f'log u: {u.log()}, -H: {-hamilton}')
            #print(theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p, sep='\n')
            return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p
        
        else:
            #print(f'inner j: {j}')
            bt_pack = self.buildtree(theta, r, u, v, j-1, epsilon)
            theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p = bt_pack
            #[print(a) for a in bt_pack]
            #print(f's_p: {s_p}')
            if s_p.sum() > 0:
                if v == -1:
                    bt_pack = self.buildtree(theta_r, r_r, u, v, j-1, epsilon)
                    theta_r, r_r, _, _, theta_pp, n_pp, s_pp = bt_pack
                    
                else:
                    bt_pack = self.buildtree(theta_f, r_f, u, v, j-1, epsilon)
                    _, _, theta_f, r_f, theta_pp, n_pp, s_pp = bt_pack
                
                update_flag = torch.rand(n_pp.size(), dtype=torch.float, 
                                         layout=n_pp.layout, device=n_pp.device)
                update_flag = update_flag < n_pp.div( n_p + n_pp )
                update_flag = torch.logical_and(update_flag, s_p.ge(1) )
                theta_p[ update_flag ] = theta_pp[ update_flag ]
                s_p = s_p * s_pp * \
                      torch.einsum('bs,bs->b', (theta_f - theta_r).flatten(1), r_r.flatten(1)) \
                        .ge(0.).type(torch.long) * \
                      torch.einsum('bs,bs->b', (theta_f - theta_r).flatten(1), r_f.flatten(1)) \
                        .ge(0.).type(torch.long)
                n_p = n_p + n_pp
            #print(theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p)
            return theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p
        
    def init_trajectory(self, theta, inertia=1.0):
        with torch.no_grad():
            r_0 = torch.randn_like( theta ).div(inertia)
            energy_0 = self.calc_energy()
            batch_dot= torch.einsum('bs,bs->b', r_0.flatten(1), r_0.flatten(1))
            hamilton = energy_0 + batch_dot.div(2.)
            u   = torch.rand_like( hamilton ).mul( torch.exp(-hamilton) )
            
            theta_r, theta_f = theta, theta
            r_r, r_f = r_0, r_0
            j = 0
            theta_m = theta
            n = torch.ones(batch_dot.size(), dtype=torch.long, layout=batch_dot.layout, device=batch_dot.device)
            s = torch.ones(batch_dot.size(), dtype=torch.long, layout=batch_dot.layout, device=batch_dot.device)
        return u, theta_r, r_r, theta_f, r_f, j, theta_m, n, s
    
    def sample_trajectory(self, theta, epsilon, inertia):
        u, theta_r, r_r, theta_f, r_f, j, theta_m, n, s = self.init_trajectory(theta, inertia)
        while (s.sum() >= 1) and (j < self.max_tree_depth):
            #print(f'on doubling {j}')
            v = torch.randn([1], dtype=torch.float, layout=theta.layout, device=theta.device) \
                  .ge(0.).mul(2.).add(-1.)
            if v < 0:
                theta_r, r_r, _, _, theta_p, n_p, s_p = self.buildtree(theta_r, r_r, u, v, j, epsilon)
            else:
                _, _, theta_f, r_f, theta_p, n_p, s_p = self.buildtree(theta_f, r_f, u, v, j, epsilon)
            
            #print('traj results:')
            #print(theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p, sep='\n')
            update_flag = torch.rand_like(n.type(torch.float))
            update_flag = update_flag <= torch.minimum( n / n_p, torch.ones_like(n.type(torch.float)) )
            update_flag = torch.logical_and( update_flag, s.ge(1) )
            update_flag = torch.logical_and( update_flag, s_p.ge(1) )
            #print(f'update_flag: {update_flag}')
            #print(update_flag)
            theta_m[ update_flag ] = theta_p[ update_flag ]
            
            n = n + n_p
            s = s * s_p * \
                torch.einsum('bs,bs->b', (theta_f - theta_r).flatten(1), r_r.flatten(1)) \
                  .ge(0.).type(torch.long) * \
                torch.einsum('bs,bs->b', (theta_f - theta_r).flatten(1), r_f.flatten(1)) \
                  .ge(0.).type(torch.long)
            #print(f's: {s}')
            j = j + 1
        
        return theta_m.detach().clone()
    
    def collect_samples(self, epsilon, inertia=1., n_samples=1, n_burnin=1):
        
        samples = []
        burnin  = []
        theta_m = self.params.theta.clone().detach()
        
        for m in range(n_burnin):
            theta_m = self.sample_trajectory( theta_m, epsilon, inertia )
            with torch.no_grad():
                self.params.theta.data = theta_m
                burnin.append( 
                    {'params':theta_m, 
                     'energy': self.calc_energy()} 
                )
        
        for m in range(n_samples):
            theta_m = self.sample_trajectory( theta_m, epsilon, inertia )
            with torch.no_grad():
                self.params.theta.data = theta_m
                samples.append( 
                    {'params':theta_m, 
                     'energy': self.calc_energy()} 
                )
        
        return {'samples': samples, 'burnin': burnin}
