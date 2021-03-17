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
    
class NUTS3(nn.Module):
    def __init__(self,
                 parameters,
                 energy_fn,
                ):
        
        super().__init__()
        self.parameters = parameters
        self.energy_fn  = energy_fn
        
        self.d_max = 1000.
        
    def leapfrog(self, grad_U, theta, r, epsilon):
                
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
            theta = theta + r.mul(epsilon)
        
        self.parameters.zero_grad()
        self.parameters.theta.data = theta
        energy = self.energy_fn(self.parameters()) \
                   .flatten(start_dim=1).sum(axis=1)
        energy.sum().backward()
        grad_U = self.parameters.theta.grad
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
        return theta, r, energy, grad_U

    def leapfrog(self, theta, r, grad_U, epsilon):
                
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
            theta = theta + r.mul(epsilon)
        
        self.parameters.theta.data = theta
        energy = self.energy_fn(self.parameters()) \
                   .flatten(start_dim=1).sum(axis=1)
        grad_U = ag.grad( energy.sum(), self.parameters.theta )[0]
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
        return theta, r, energy, grad_U
    
    def calc_energy(self):
        energy = self.energy_fn(self.parameters())
        energy = self.parameters.rebatch( energy )
        return energy

    def leapfrog(self, theta, r, epsilon):
        
        self.parameters.theta.data = theta
        energy = calc_energy()
        grad_U = ag.grad( energy.sum(), self.parameters.theta )[0]
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
            theta = theta + r.mul(epsilon)
            
        self.parameters.theta.data = theta
        energy = calc_energy()
        grad_U = ag.grad( energy.sum(), self.parameters.theta )[0]
        
        with torch.no_grad():
            r = r - grad_U.mul(epsilon).div(2.)
            
        return theta, r, energy
        
    def buildtree(self, theta, r, u, v, j, epsilon):
        if j == 0:
            theta_p, r_p, energy_p = self.leapfrog(theta, r, epsilon)
            batch_dot = torch.einsum('bs,bs->b', r_p.flatten(1), r_p.flatten(1))
            hamilton  = energy_p + batch_dot.div(2.)
            n_p = (u <= torch.exp(-hamilton)).type(torch.long)
            s_p = (torch.log(u).add(-self.d_max) < -hamilton).type(torch.long)
            return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p
        
        else:
            bt_pack = self.buildtree(self, theta, r, u, v, j-1, epsilon)
            theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p = bt_pack
            if s_p.sum() > 0:
                if v == -1:
                    bt_pack = self.buildtree(self, theta_r, r_r, u, v, j-1, epsilon)
                    theta_r, r_r, _, _, theta_pp, n_pp, s_pp = bt_pack
                    
                else:
                    bt_pack = self.buildtree(self, theta_f, r_f, u, v, j-1, epsilon)
                    _, _, theta_f, r_f, theta_pp, n_pp, s_pp = bt_pack
                    
                update_flag = torch.rand_like(n_pp) < n_pp.div( n_p + n_pp )
                update_flag = torch.logical_and(update_flag, s_p.ge(1) )
                theta_p[ update_flag ] = theta_pp[ update_flag ]
                s_p = s_p * s_pp * \
                      torch.einsum('bs,bs->b', (theta_f - theta_r), r_r).ge(0.).type(torch.long) * \
                      torch.einsum('bs,bs->b', (theta_f - theta_r), r_f).ge(0.).type(torch.long)
                n_p = n_p + n_pp
                
            return theta_r, r_r, theta_f, r_f, theta_p, n_p, s_p
        
    def init_trajectory(self, theta):
        with torch.no_grad():
            r_0 = torch.randn_like( theta )
            energy_0 = self.calc_energy()
            batch_dot= torch.einsum('bs,bs->b', r_0.flatten(1), r_p.flatten(1))
            hamilton = energy_0 + batch_dot.div(2.)
            u   = torch.rand_like( hamilton ).mul( torch.exp(-hamilton) )
            
            theta_r, theta_f = theta, theta
            r_r, r_f = r_0, r_0
            j = 0
            theta_m = theta
            n = torch.ones(batch_dot.size(), dtype=torch.long, layout=batch_dot.layout, device=batch_dot.device)
            s = torch.ones(batch_dot.size(), dtype=torch.long, layout=batch_dot.layout, device=batch_dot.device)
        return u, theta_r, r_r, theta_f, r_f, j, theta_m, n, s
    
    def sample_trajectory(self, theta, epsilon):
        u, theta_r, r_r, theta_f, r_f, j, theta_m, n, s = self.init_trajectory(theta)
        while s.sum() >= 1:
            v = torch.randn([1], dtype=torch.float, layout=theta.layout, device=theta.device) \
                  .ge(0.).mul(2.).add(-1.)
            if v < 0:
                theta_r, r_r, _, _, theta_p, n_p, s_p = self.buildtree(theta_r, r_r, u, v, j, epsilon)
            else:
                _, _, theta_f, r_f, theta_p, n_p, s_p = self.buildtree(theta_f, r_f, u, v, j, epsilon)
            
            update_flag = torch.minimum( n / n_p, torch.ones_like(n.type(torch.float)) ) <= torch.rand_like(n)
            update_flag = torch.logical_and( update_flag, s.ge(1) )
            update_flag = torch.logical_and( update_flag, s_p.ge(1) )
            theta_m[ update_flag ] = theta_p[ update_flag ]
            
            n = n + n_p
            s = s * s_p * \
                torch.einsum('bs,bs->b', (theta_f - theta_r), r_r).ge(0.).type(torch.long) * \
                torch.einsum('bs,bs->b', (theta_f - theta_r), r_f).ge(0.).type(torch.long)
            j = j + 1
        
        return theta_m.detach().clone()
    
    def collect_samples(self, epsilon, n_samples=1):
        samples = []
        theta_m = self.params.theta.clone().detach()
        for m in range(n_samples):
            theta_m = self.sample_trajectory( theta_m, epsilon )
            samples.append( theta_m )
        return samples
            

        

