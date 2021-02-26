import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import sys
sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
from boda.common import utils 


class NUTS_parameters(nn.Module):
    def __init__(self,
                 theta_0=None,
                 num_sequences=1,
                 num_st_samples=1,
                 seq_len=200, 
                 padding_len=400,
                 vocab_len=4,
                 temperature=1,
                 ST_sampling=True,
                 **kwargs):
        """
        

        Parameters
        ----------
        theta_0 : torch tensor, optional
            DESCRIPTION. The default is None.
        num_sequences : int, optional
            DESCRIPTION. The default is 1.
        num_st_samples : int, optional
            DESCRIPTION. The default is 1.
        seq_len : int, optional
            DESCRIPTION. The default is 200.
        padding_len : int, optional
            DESCRIPTION. The default is 400.
        vocab_len : int, optional
            DESCRIPTION. The default is 4.
        temperature : float, optional
            DESCRIPTION. The default is 1.
        ST_sampling : bool, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(NUTS_parameters, self).__init__()
        self.theta_0 = theta_0
        self.num_sequences = num_sequences
        self.num_st_samples = num_st_samples
        self.seq_len = seq_len  
        self.padding_len = padding_len
        self.vocab_len = vocab_len
        self.temperature = temperature
        self.ST_sampling = ST_sampling
        
        self.softmax = nn.Softmax(dim=1)
        self.grad = torch.autograd.grad
        
        # register initial theta. if it was not given, initialize randomly
        self.initialize_theta(one_hot=True)
        
        # initialize the momentum r
        self.register_buffer('r', torch.randn_like(self.theta))
        
        # create and register the padding tensors
        upPad_logits, downPad_logits = utils.create_paddingTensors(self.num_sequences, self.padding_len,
                                                                   self.num_st_samples, self.ST_sampling)
        self.register_buffer('upPad_logits', upPad_logits)
        self.register_buffer('downPad_logits', downPad_logits)
                
    def forward(self, theta):
        softmaxed_theta = self.softmax(theta / self.temperature)
        if self.ST_sampling:
           softmaxed_theta = self.softmax(theta / self.temperature)
           probs = Categorical(torch.transpose(softmaxed_theta, 1, 2))
           idxs = probs.sample((self.num_st_samples, ))
           sampled_theta_T = F.one_hot(idxs, num_classes=self.vocab_len)   
           sampled_theta = torch.transpose(sampled_theta_T, 2, 3)
           sampled_theta = sampled_theta - softmaxed_theta.detach() + softmaxed_theta
           sampled_theta = self.pad(sampled_theta)
           sampled_theta = sampled_theta.view(self.num_st_samples * self.num_sequences, self.vocab_len, -1)
           return sampled_theta
        else:
            return self.pad(softmaxed_theta)
         
    def initialize_theta(self, one_hot=True):
        if self.theta_0 is not None:
            self.register_buffer('theta', self.theta_0)
        else:
            size = (self.num_sequences, self.vocab_len, self.seq_len)
            if one_hot:
                theta = np.zeros(size)
                for seqIdx in range(self.num_sequences):
                    for step in range(self.seq_len):
                        random_token = np.random.randint(self.vocab_len)
                        theta[seqIdx, random_token, step] = 1      
                self.register_buffer('theta', torch.tensor(theta, dtype=torch.float))
            else:
                self.register_buffer('theta', self.softmax(torch.rand(size)))
           
    def pad(self, tensor, multi_sampling_off=False):
        if self.padding_len > 0:
            if multi_sampling_off:
                size = tensor.shape
                upPad_logits, downPad_logits = self.upPad_logits[0].view(size), self.downPad_logits[0].view(size)
                return torch.cat([ upPad_logits, tensor, downPad_logits], dim=-1)
            else:
                return torch.cat([ self.upPad_logits, tensor, self.downPad_logits], dim=-1)
        else: 
            return tensor
        
        
class NUTS6(nn.Module):
    def __init__(self,
                 phase_params,
                 fitness_fn,
                 kinetic_scale_factor=1,
                 **kwargs):
        """
        

        Parameters
        ----------
        phase_params : torch Module
            DESCRIPTION.
        fitness_fn : torch Module
            DESCRIPTION.
        kinetic_scale_factor : float, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(NUTS6, self).__init__()
        self.phase_params = phase_params
        self.fitness_fn = fitness_fn
        self.kinetic_scale_factor = kinetic_scale_factor
        self.Delta_max = 1000
        self.delta = 0.65
        self.fitness_hist = []
        
        self.softmax = nn.Softmax(dim=1)
        self.grad = torch.autograd.grad
    
    def run(self, M, M_adapt, max_height=10):  
        self.M = M
        self.M_adapt = M_adapt
        epsilon = self.find_reasonable_epsilon()
        #print('Initial distributions:')
        #print(self.phase_params.theta)
        mu = np.log(10 * epsilon)
        epsilon_bar = 1
        H_bar = 0.
        gamma = 0.05
        t_0 = 10
        kappa = 0.75
        initial_fitness = self.fitness_fn(self.phase_params.pad(self.phase_params.theta, multi_sampling_off=True)).mean()
        self.fitness_hist.append(initial_fitness.item())
        visited_theta = []
        visited_theta.append(self.phase_params.theta.numpy())
        pbar = tqdm(range(self.M), desc='NUTS Trials')
        for m in pbar:
            pbar.set_postfix({'epsilon': epsilon})
            r_0 = torch.randn_like(self.phase_params.r)
            #changed 0 to 1e-15 to avoid possible log(0) in build_tree
            u = np.random.uniform(1e-15, self.p_fn(theta=self.phase_params.theta, r=r_0))
            theta_minus, theta_plus = self.phase_params.theta, self.phase_params.theta
            r_minus, r_plus = self.phase_params.r, self.phase_params.r
            j, n, s = 0, 1, 1
            while s == 1 and j <= max_height:
                v = np.random.choice([-1, 1])
                if v == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_minus, r_minus, u, v, j, epsilon, self.phase_params.theta, r_0)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_plus, r_plus, u, v, j, epsilon, self.phase_params.theta, r_0)
                if s_prime == 1:
                    random_prob = float(np.random.rand(1))
                    if min(1, n_prime / n) >= random_prob:
                        self.phase_params.theta = theta_prime
                        visited_theta.append(theta_prime)
                n += n_prime
                s = s_prime * self.stop_indicator(theta_minus, r_minus, theta_plus, r_plus)
                j += 1
            if m < self.M_adapt:
                H_bar = (1 - 1/(m + t_0)) * H_bar + 1/(m + t_0) * (self.delta - alpha / n_alpha)
                epsilon = np.exp(mu - H_bar * np.sqrt(m+1) / gamma)
                #re-wrote: log(epsilon_bar) = m ** (-kappa) * log(epsilon) + ( 1 - m ** (-kappa)) * log(epsilon_bar)
                epsilon_bar = epsilon_bar * (epsilon * epsilon_bar) ** (1 / (m+1) ** kappa)
        self.visited_theta = visited_theta

            
    def L_fn(self, theta):
        return -self.fitness_fn(self.phase_params(theta)).mean()
    
    def p_fn(self, theta=None, r=None, L=None):
        if theta is not None:
            return torch.exp(self.L_fn(theta) - r.pow(2).sum()/2 * self.kinetic_scale_factor).item()
        elif L is not None:
            return torch.exp(L - r.pow(2).sum()/2 * self.kinetic_scale_factor).item()
        
    def leapfrog(self, theta, r, epsilon):
        # make theta a leaf
        theta.requires_grad_()    
        # compute grad
        L = self.L_fn(theta)
        L_grad = self.grad(L, theta, retain_graph=False)[0]        
        # momentum half step
        r_prime = r + 0.5 * epsilon * L_grad        
        # position full step
        theta_prime = theta + epsilon * r_prime       
        # compute new grad
        L_prime = self.L_fn(theta_prime)
        L_grad_prime = self.grad(L_prime, theta_prime, retain_graph=False)[0]        
        # momentum half step
        r_prime = r_prime + 0.5 * epsilon * L_grad_prime 
        p_prime = self.p_fn(r=r_prime, L=L_prime)
        # return non-leaf
        theta_prime = theta_prime.detach()
        theta.requires_grad = False
        return theta_prime, r_prime, L_prime.item(), L_grad_prime, p_prime
    
    # adapted from https://github.com/mfouesneau/NUTS
    def find_reasonable_epsilon(self):
        p = self.p_fn(theta=self.phase_params.theta, r=self.phase_params.r)
        epsilon = 1
        _, r_prime, _, L_grad_prime, p_prime = self.leapfrog(self.phase_params.theta, self.phase_params.r, epsilon)
        # check the initial step size does not yield infinite values of p or the grad
        k = 1
        while np.isinf(p_prime) or torch.isinf(L_grad_prime).any():   
            k *= 0.5
            epsilon = k * epsilon
            _, _, _, L_grad_prime, p_prime = self.leapfrog(self.phase_params.theta, self.phase_params.r, epsilon)              
        # set a = 2*I[p_prime/p > 0.5] - 1
        a = 1. if p_prime/(p + 1e-12) > 0.5 else -1.
        while (np.log(p_prime + 1e-12) - np.log(p + 1e-12)) * a > np.log(0.5) * a:
            epsilon = epsilon * (2. ** a)
            _, _, _, _, p_prime = self.leapfrog(self.phase_params.theta, self.phase_params.r, epsilon)
        #print(f'Initial reasonable epsilon = {epsilon}')
        return epsilon
    
    def stop_indicator(self, theta_minus, r_minus, theta_plus, r_plus):
        theta_diff = theta_plus - theta_minus
        s = (torch.mul(theta_diff, r_plus).sum() >= 0) and (torch.mul(theta_diff, r_minus).sum() >= 0)
        if s == False: print('Trial stopped by U-turn')
        return 1 * s.item()
        
    def build_tree(self, theta, r, u, v, j, epsilon, theta_0, r_0):
        if j == 0:   
            theta_prime, r_prime, L_prime, L_grad_prime, p_prime = self.leapfrog(theta, r, v * epsilon)
            n_prime = 1 if u <= p_prime else 0
            s_prime = 1 if np.log(u) < (np.log(p_prime + 1e-12) + self.Delta_max) else 0
            p_0 = self.p_fn(theta=theta_0, r=r_0)
            alpha = min(1, p_prime/(p_0 + 1e-12))
            self.fitness_hist.append(-L_prime)
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha, 1.
        else:
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self.build_tree(theta, r, u, v, j-1, epsilon, theta_0, r_0)
            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self.build_tree(theta_minus, r_minus, u, v, j-1, epsilon, theta_0, r_0)
                else:
                    _, _, theta_plus, r_plus, theta_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                self.build_tree(theta_plus, r_plus, u, v, j-1, epsilon, theta_0, r_0)
                random_prob = float(np.random.rand(1))
                if (n_prime / max(1, n_prime + n_2prime)) >= random_prob:
                    theta_prime = theta_2prime
                alpha_prime += alpha_2prime
                n_alpha_prime += n_alpha_2prime
                s_prime = s_2prime * self.stop_indicator(theta_minus, r_minus, theta_plus, r_plus)
                n_prime += n_2prime
            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        
    def plots(self):
        if self.phase_params.num_sequences > 1:
            fitness_plot_title = f'Fitness-average history after {self.M} trials'
        else:
            fitness_plot_title = f'Fitness history after {self.M} trials'
        plt.plot(self.fitness_hist, linestyle="",marker=".")
        plt.title(fitness_plot_title)
        plt.show()      
           
            
        

#--------------------------- EXAMPLE ----------------------------------------
if __name__ == '__main__':
    
    phase_params = NUTS_parameters(theta_0=None,
                                num_sequences=1,
                                num_st_samples=10,
                                temperature=1,
                                ST_sampling=True)  
    sampler = NUTS6(phase_params=phase_params,
                    fitness_fn=utils.first_token_rewarder,
                    kinetic_scale_factor=0.028)    
    sampler.run(M=20, M_adapt=10, max_height=10)  
    sampler.plots()
