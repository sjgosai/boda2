import sys
import time
import warnings
import math

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
    """
    Base class for implementing Leapfrog integration and related methods.

    Methods:
        calc_energy(T=1.0):
            Calculate the energy of the system.
        leapfrog(theta, r, epsilon, T=1.0):
            Perform a single Leapfrog integration step.
        init_eta(epsilon, inertia=1.):
            Initialize step sizes for Leapfrog integration.
        eval_samples(sample_tensor):
            Evaluate the negative log-likelihood of samples.

    """
    
    def __init__(self):
        """
        Initialize the LeapfrogBase module.
        """
        super().__init__()
        
    def calc_energy(self, T=1.0):
        """
        Calculate the energy of the system.

        Args:
            T (float, optional): Temperature parameter. Default is 1.0.

        Returns:
            torch.Tensor: Calculated energy values.
        """
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
        """
        Perform a single Leapfrog integration step.

        Args:
            theta (torch.Tensor): Current position.
            r (torch.Tensor): Current momentum.
            epsilon (torch.Tensor): Step size.
            T (float, optional): Temperature parameter. Default is 1.0.

        Returns:
            tuple: Tuple containing updated position, momentum, energy, and gradient of the energy.
        """
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
    
    def init_eta(self, epsilon, inertia=1.):
        """
        Initialize step sizes for Leapfrog integration.

        Args:
            epsilon (torch.Tensor): Initial step size.
            inertia (float, optional): Inertia factor. Default is 1.

        Returns:
            torch.Tensor: Initialized step sizes.
        """
        theta_0 = self.params.theta.data.clone().detach()
        energy_0= self.calc_energy()
        r_0     = torch.randn_like( theta_0 ).div(inertia)
        epsilon = epsilon.to(r_0.device)
        nll_0 = energy_0 + r_0.pow(2).flatten(1).sum(1).div(2.)
        theta_p, r_p, energy_p, grad_U = self.leapfrog(theta_0, r_0, epsilon, T=1.0)
        nll_p = energy_p + r_p.pow(2).flatten(1).sum(1).div(2.)
        
        a = (nll_0 - nll_p).exp().ge(0.5).float().mul(2.).add(-1)
        s = (nll_0 - nll_p).exp().pow(a) > a.mul(-math.log(2.)).exp()
        
        while s.sum() > 0:
            proposal = a.mul(math.log(2.)).exp() * epsilon.squeeze()
            epsilon[s] = proposal[s].view(*epsilon[s].shape)
            theta_p, r_p, energy_p, grad_U = self.leapfrog(theta_0, r_0, epsilon, T=1.0)
            nll_p = energy_p + r_p.pow(2).flatten(1).sum(1).div(2.)
            s = (nll_0 - nll_p).exp().pow(a) > a.mul(-math.log(2.)).exp()
            
        return epsilon

    def eval_samples(self, sample_tensor):
        """
        Evaluate the negative log-likelihood of samples.

        Args:
            sample_tensor (torch.Tensor): Tensor containing samples.

        Returns:
            torch.Tensor: Negative log-likelihood values for the samples.
        """
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
    """
    Base class for implementing temperature schedules.

    Methods:
        step(inner=None, outer=None):
            Perform a temperature schedule step.
        forward():
            Calculate the current temperature.

    """
    
    def __init__(self):
        """
        Initialize the TemperatureScheduleBase module.
        """
        super().__init__()
        
    def step(inner=None, outer=None):
        """
        Perform a temperature schedule step.

        Args:
            inner (int, optional): Inner loop iteration count. Default is None.
            outer (int, optional): Outer loop iteration count. Default is None.

        Returns:
            None
        """
        return None
        
    def forward():
        """
        Calculate the current temperature.

        Returns:
            float: The current temperature.
        """
        return 1.0    

class CosineAnnealingSchedule(TemperatureScheduleBase):
    """
    Temperature schedule based on the cosine annealing strategy.

    Args:
        T_max (int): Maximum number of iterations in a cycle.
        eta_max (float or torch.Tensor): Maximum temperature value.
        eta_min (float or torch.Tensor, optional): Minimum temperature value. Defaults to None.

    Methods:
        step(value=None):
            Perform a step in the temperature schedule.

    """
    
    def __init__(self, T_max, eta_max, eta_min=None):
        """
        Initialize the CosineAnnealingSchedule module.

        Args:
            T_max (int): Maximum number of iterations in a cycle.
            eta_max (float or torch.Tensor): Maximum temperature value.
            eta_min (float or torch.Tensor, optional): Minimum temperature value. Defaults to None.
        """
        super().__init__()
        
        self.T_max = T_max
        self.eta_max = eta_max
        if eta_min is None:
            self.eta_min = torch.zeros_like(eta_max)
        else:
            self.eta_min = eta_min
            
        self._step = 0
        
    def step(self, value=None):
        """
        Perform a step in the temperature schedule.

        Args:
            value (int, optional): Value to update the step counter. If None, the step counter is incremented by 1. Defaults to None.

        Returns:
            dict: Dictionary containing the updated '_step' value.
        """
        if value is None:
            self._step += 1
        else:
            self._step = value % (self.T_max+1)
            
        return {'_step':self._step}
    
    def forward(self):
        """
        Compute the temperature value based on the cosine annealing schedule.

        Returns:
            torch.Tensor: Computed temperature value.
        """
        hook = torch.tensor( math.pi * self._step / self.T_max )
        hook = torch.cos( hook ) + 1
        hook = (self.eta_max - self.eta_min).mul(hook).div(2.)
        return hook + self.eta_min
        
class LogCosineSchedule(CosineAnnealingSchedule):
    """
    Temperature schedule based on the logarithmically transformed cosine annealing strategy.

    Args:
        T_max (int): Maximum number of iterations in a cycle.
        eta_max (float or torch.Tensor): Maximum temperature value.
        eta_min (float or torch.Tensor, optional): Minimum temperature value. Defaults to None.

    Methods:
        step(value=None):
            Perform a step in the temperature schedule.

        forward():
            Compute the temperature value based on the logarithmically transformed cosine annealing schedule.

    """
    
    def __init__(self, T_max, eta_max, eta_min=None):
        """
        Initialize the LogCosineSchedule module.

        Args:
            T_max (int): Maximum number of iterations in a cycle.
            eta_max (float or torch.Tensor): Maximum temperature value.
            eta_min (float or torch.Tensor, optional): Minimum temperature value. Defaults to None.
        """
        super().__init__(T_max, eta_max, eta_min)
        
    def forward(self):
        """
        Compute the temperature value based on the logarithmically transformed cosine annealing schedule.

        Returns:
            torch.Tensor: Computed temperature value after logarithmic transformation.
        """
        hook = torch.tensor( math.pi * self._step / self.T_max )
        hook = torch.cos( hook ) + 1
        hook = (self.eta_max - self.eta_min).mul(hook).div(2.)
        return torch.exp( hook + self.eta_min )
            
class WaveSchedule(TemperatureScheduleBase):
    """
    Temperature schedule based on the wave-like annealing strategy.

    Args:
        T_init (float, optional): Initial temperature value. Defaults to 1.0.
        T_max (float, optional): Maximum temperature value. Defaults to None.
        T_min (float, optional): Minimum temperature value. Defaults to None.
        outer_warmup (int, optional): Number of outer warmup steps. Defaults to None.
        outer_cooldown (int, optional): Number of outer cooldown steps. Defaults to None.
        inner_warmup (int, optional): Number of inner warmup steps. Defaults to None.
        inner_cooldown (int, optional): Number of inner cooldown steps. Defaults to None.

    Methods:
        step(inner=None, outer=None):
            Perform a step in the temperature schedule.

        get_inner_T():
            Compute the inner temperature value based on the wave-like annealing schedule.

        get_outer_T():
            Compute the outer temperature value based on the wave-like annealing schedule.

        forward():
            Compute the temperature value based on the wave-like annealing schedule.

    """
    
    def __init__(self, T_init=1.0, T_max=None, T_min=None, 
                 outer_warmup=None, outer_cooldown=None, 
                 inner_warmup=None, inner_cooldown=None):
        """
        Initialize the WaveSchedule module.

        Args:
            T_init (float, optional): Initial temperature value. Defaults to 1.0.
            T_max (float, optional): Maximum temperature value. Defaults to None.
            T_min (float, optional): Minimum temperature value. Defaults to None.
            outer_warmup (int, optional): Number of outer warmup steps. Defaults to None.
            outer_cooldown (int, optional): Number of outer cooldown steps. Defaults to None.
            inner_warmup (int, optional): Number of inner warmup steps. Defaults to None.
            inner_cooldown (int, optional): Number of inner cooldown steps. Defaults to None.
        """
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
        """
        Perform a step in the temperature schedule.

        Args:
            inner (int, optional): Inner step count. Defaults to None.
            outer (int, optional): Outer step count. Defaults to None.

        Returns:
            dict: Dictionary containing the updated step counts.
        """
        if inner is not None:
            self.inner_step = inner
            
        if outer is not None:
            self.outer_step = outer
            
        return {'outer_step':self.outer_step, 
                'inner_step':self.inner_step}
    
    def get_inner_T(self):
        """
        Compute the inner temperature value based on the wave-like annealing schedule.

        Returns:
            float: Computed inner temperature value.
        """
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
        """
        Compute the outer temperature value based on the wave-like annealing schedule.

        Returns:
            float: Computed outer temperature value.
        """
        if self.outer_step < self.outer_warmup:
            hook = self.T_init + (self.outer_step/self.outer_warmup)
        elif self.outer_step < self.outer_warmup+self.outer_cooldown:
            hook = (1+self.T_init)*(1 - ((self.outer_step-self.outer_warmup)/self.outer_cooldown))
        else:
            hook = 0.
        return hook
    
    def forward(self):
        """
        Compute the temperature value based on the wave-like annealing schedule.

        Returns:
            float: Computed temperature value after wave-like annealing.
        """
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
    """
    Gradient Descent Test module using the Leapfrog integrator.

    Args:
        params: Parameter object representing the model's parameters.
        energy_fn: Energy function that computes the energy given model parameters.

    Methods:
        collect_samples(epsilon, n_samples=1):
            Collect samples using the Leapfrog integrator and gradient descent.

    """
    
    def __init__(self,
                 params,
                 energy_fn
                ):
        """
        Initialize the GDTest module.

        Args:
            params: Parameter object representing the model's parameters.
            energy_fn: Energy function that computes the energy given model parameters.
        """
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
    def collect_samples(self, epsilon, n_samples=1):
        """
        Collect samples using the Leapfrog integrator and gradient descent.

        Args:
            epsilon: Step size for the Leapfrog integration.
            n_samples (int, optional): Number of samples to collect. Defaults to 1.

        Returns:
            dict: Dictionary containing the collected samples, energies, and gradients.
        """
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
    """
    Hamiltonian Monte Carlo (HMC) sampler using the Leapfrog integrator.

    Args:
        params: Parameter object representing the model's parameters.
        energy_fn: Energy function that computes the energy given model parameters.

    Methods:
        sample_trajectory(theta, epsilon, L, inertia=1., alpha=1.):
            Sample a trajectory using the Leapfrog integrator and HMC.
        collect_samples(epsilon, L, inertia=1., alpha=1., n_samples=1, n_burnin=0):
            Collect samples using the HMC sampler.

    """
    
    def __init__(self,
                 params,
                 energy_fn
                ):
        """
        Initialize the HMC sampler.

        Args:
            params: Parameter object representing the model's parameters.
            energy_fn: Energy function that computes the energy given model parameters.
        """
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
        
    def sample_trajectory(self, theta, epsilon, L, inertia=1., alpha=1.):
        """
        Sample a trajectory using the Leapfrog integrator and Hamiltonian Monte Carlo (HMC).

        Args:
            theta: Initial parameter values.
            epsilon: Step size for the Leapfrog integration.
            L: Number of Leapfrog steps.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.

        Returns:
            tuple: Tuple containing the sampled parameters, energy, and acceptance information.
        """
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
        """
        Collect samples using the Hamiltonian Monte Carlo (HMC) sampler.

        Args:
            epsilon: Step size for the Leapfrog integration.
            L: Number of Leapfrog steps.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.
            n_samples (int, optional): Number of samples to collect. Defaults to 1.
            n_burnin (int, optional): Number of burn-in samples. Defaults to 0.

        Returns:
            dict: Dictionary containing the collected samples and burn-in samples.
        """
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
            jitter = torch.rand_like(epsilon).mul(0.2).add(0.9)
            theta_m, U_m, accept_m = self.sample_trajectory( theta_m, epsilon.mul(jitter), L, inertia, alpha )
            with torch.no_grad():
                samples.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach()} 
                )
            
        return {'samples': samples, 'burnin': burnin}

class HMCDA(LeapfrogBase):
    """
    Hamiltonian Monte Carlo (HMC) sampler with Dual Averaging adaptation.

    Args:
        params: Parameter object representing the model's parameters.
        energy_fn: Energy function that computes the energy given model parameters.

    Methods:
        sample_trajectory(theta, epsilon, L, inertia=1., alpha=1.):
            Sample a trajectory using the Leapfrog integrator and HMC with Dual Averaging.
        collect_samples(epsilon_base, inertia=1., alpha=1., n_samples=1, n_burnin=1, delta=0.65, lambd=1.0, gamma=0.05, kappa=0.75, t_0=10):
            Collect samples using the HMC sampler with Dual Averaging adaptation.

    """
    
    def __init__(self,
                 params,
                 energy_fn
                ):
        """
        Initialize the HMCDA sampler.

        Args:
            params: Parameter object representing the model's parameters.
            energy_fn: Energy function that computes the energy given model parameters.
        """
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
    def sample_trajectory(self, theta, epsilon, L, inertia=1., alpha=1.):
        """
        Sample a trajectory using the Leapfrog integrator and Hamiltonian Monte Carlo (HMC)
        with Dual Averaging adaptation.

        Args:
            theta: Initial parameter values.
            epsilon: Step size for the Leapfrog integration.
            L: Number of Leapfrog steps.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.

        Returns:
            tuple: Tuple containing the sampled parameters, energy, acceptance information,
                   and acceptance probabilities.
        """
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
            
            accept_prob = (c_U - p_U + c_K - p_K).exp().clamp(max=1.)
            
            accept = torch.rand_like(c_U) < accept_prob
            
            theta_p = torch.stack([theta_0, theta], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            U       = torch.stack([c_U, p_U], dim=0) \
                        [accept.long(), torch.arange(c_U.numel())]
            
            return theta_p, U, accept, accept_prob
        
    def collect_samples(self,
                        epsilon_base, 
                        inertia=1., 
                        alpha=1., 
                        n_samples=1, 
                        n_burnin=1, 
                        delta=0.65, 
                        lambd=1.0, 
                        gamma=0.05, 
                        kappa=0.75, 
                        t_0=10
                       ):
        """
        Collect samples using the Hamiltonian Monte Carlo (HMC) sampler with Dual Averaging adaptation.

        Args:
            epsilon_base: Base step size for the Leapfrog integration.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.
            n_samples (int, optional): Number of samples to collect. Defaults to 1.
            n_burnin (int, optional): Number of burn-in samples. Defaults to 1.
            delta (float, optional): Target acceptance probability. Defaults to 0.65.
            lambd (float, optional): Scaling factor for Leapfrog step size. Defaults to 1.0.
            gamma (float, optional): Scaling factor for the Dual Averaging adaptation. Defaults to 0.05.
            kappa (float, optional): Scaling factor for Dual Averaging adaptation. Defaults to 0.75.
            t_0 (int, optional): Tuning parameter for Dual Averaging adaptation. Defaults to 10.

        Returns:
            dict: Dictionary containing the collected samples and burn-in samples.
        """
        epsilon = self.init_eta(epsilon_base, inertia=inertia)
        eps_bar = epsilon_base.to(epsilon.device)
        mu = epsilon.mul(10.).log()
        
        samples = []
        burnin  = []
        theta_m = self.params.theta.clone().detach()
        H_bar   = torch.zeros_like( self.calc_energy() )
        
        for m in tqdm(range(n_burnin)):
            L = epsilon.div(lambd).pow(-1).round().clamp(min=1.).max().long().item()
            print(f'Leapfrog steps: {L}')
            print(f'Epsilon steps: {epsilon}')
            theta_m, U_m, accept_m, accept_prob = self.sample_trajectory( theta_m, epsilon, L, inertia, alpha )
            print(f'Accept probs: {accept_prob}')
            
            H_update = accept_prob.add(-delta).mul(-1).div(m + 1 + t_0)
            H_old    = H_bar.mul( 1 - ( 1 / (m + 1 + t_0) ) )
            H_bar    = H_old + H_update
            
            epsilon = H_bar.view( *mu.shape ).mul( ((m+1)**0.5)/gamma ).mul(-1).add(mu).exp()
            eps_bar = eps_bar.log().mul( 1 - ((m+1)**-kappa) ) + epsilon.log().mul( (m+1)**-kappa )
            eps_bar = eps_bar.exp()
            
            with torch.no_grad():
                burnin.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': epsilon.clone().detach(),
                     'steps': L} 
                )

        L = eps_bar.div(lambd).pow(-1).round().clamp(min=1.).max().long().item()
        for m in tqdm(range(n_samples)):
            jitter = torch.rand_like(eps_bar).mul(0.2).add(0.9)
            theta_m, U_m, accept_m, accept_prob = self.sample_trajectory( theta_m, eps_bar.mul(jitter), L, inertia, alpha )
            with torch.no_grad():
                samples.append( 
                    {'params':theta_m, 
                     'energy': U_m, 
                     'acceptance': accept_m, 
                     'epsilon': eps_bar.clone().detach(),
                     'steps': L} 
                )
            
        return {'samples': samples, 'burnin': burnin}

    
class AnnealingHMC(LeapfrogBase):
    """
    Annealing Hamiltonian Monte Carlo (HMC) sampler with temperature schedule.

    Args:
        params: Parameter object representing the model's parameters.
        energy_fn: Energy function that computes the energy given model parameters.
        temperature_schedule: Temperature schedule for annealing.

    Methods:
        sample_trajectory(theta, epsilon, L, inertia=1., alpha=1.):
            Sample a trajectory using the Leapfrog integrator and Annealing HMC.
        collect_samples(epsilon, L, inertia=1., alpha=1., n_samples=1, n_burnin=0):
            Collect samples using the Annealing HMC sampler.

    """
    
    def __init__(self,
                 params,
                 energy_fn,
                 temperature_schedule
                ):
        """
        Initialize the AnnealingHMC sampler.

        Args:
            params: Parameter object representing the model's parameters.
            energy_fn: Energy function that computes the energy given model parameters.
            temperature_schedule: Temperature schedule for annealing.
        """
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        self.temperature_schedule = temperature_schedule
        
    def sample_trajectory(self, theta, epsilon, L, inertia=1., alpha=1.):
        """
        Sample a trajectory using the Leapfrog integrator and Annealing Hamiltonian Monte Carlo (HMC).

        Args:
            theta: Initial parameter values.
            epsilon: Step size for the Leapfrog integration.
            L: Number of Leapfrog steps.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.

        Returns:
            tuple: Tuple containing the sampled parameters, energy, and acceptance information.
        """
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
        """
        Collect samples using the Annealing Hamiltonian Monte Carlo (HMC) sampler.

        Args:
            epsilon: Step size for the Leapfrog integration.
            L: Number of Leapfrog steps.
            inertia: Inertia for initializing momentum. Defaults to 1.
            alpha: Scaling factor for momentum rescaling. Defaults to 1.
            n_samples (int, optional): Number of samples to collect. Defaults to 1.
            n_burnin (int, optional): Number of burn-in samples. Defaults to 0.

        Returns:
            dict: Dictionary containing the collected samples and burn-in samples.
        """
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
    """
    No-U-Turn Sampler (NUTS) with generalized tree doubling and shrinking criterion.

    Args:
        params: Parameter object representing the model's parameters.
        energy_fn: Energy function that computes the energy given model parameters.
        max_tree_depth: Maximum depth of the binary tree. Defaults to 10.

    Methods:
        buildtree(theta, r, u, v, j, epsilon):
            Build a trajectory for the NUTS sampler.
        init_trajectory(theta, inertia=1.0):
            Initialize a trajectory for the NUTS sampler.
        sample_trajectory(theta, epsilon, inertia):
            Sample a trajectory using the NUTS sampler.
        collect_samples(epsilon, inertia=1., n_samples=1, n_burnin=1):
            Collect samples using the NUTS sampler.

    """
    
    def __init__(self,
                 params,
                 energy_fn,
                 max_tree_depth=10
                ):
        """
        Initialize the NUTS3 sampler.

        Args:
            params: Parameter object representing the model's parameters.
            energy_fn: Energy function that computes the energy given model parameters.
            max_tree_depth: Maximum depth of the binary tree. Defaults to 10.
        """
        super().__init__()
        self.params = params
        self.energy_fn  = energy_fn
        self.max_tree_depth = max_tree_depth
        
        self.d_max = 1000.
        
    def buildtree(self, theta, r, u, v, j, epsilon):
        """
        Build a trajectory for the NUTS sampler.

        Args:
            theta: Current parameter values.
            r: Current momentum values.
            u: Random uniform values for sampling.
            v: Direction of tree doubling.
            j: Depth of the binary tree.
            epsilon: Step size for the Leapfrog integration.

        Returns:
            tuple: Tuple containing the trajectory elements for the NUTS sampler.
        """
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
        """
        Initialize a trajectory for the NUTS sampler.

        Args:
            theta: Initial parameter values.
            inertia: Inertia for initializing momentum. Defaults to 1.0.

        Returns:
            tuple: Tuple containing the initialized trajectory elements for the NUTS sampler.
        """
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
        """
        Sample a trajectory using the NUTS sampler.

        Args:
            theta: Initial parameter values.
            epsilon: Step size for the Leapfrog integration.
            inertia: Inertia for initializing momentum.

        Returns:
            torch.Tensor: Sampled parameters using the NUTS sampler.
        """
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
        """
        Collect samples using the NUTS sampler.

        Args:
            epsilon: Step size for the Leapfrog integration.
            inertia: Inertia for initializing momentum. Defaults to 1.0.
            n_samples: Number of samples to collect. Defaults to 1.
            n_burnin: Number of burn-in samples. Defaults to 1.

        Returns:
            dict: Dictionary containing the collected samples and burn-in samples.
        """
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
