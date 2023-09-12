import argparse
import sys
import copy

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from ..common import constants, utils

def mask_gradients(in_tensor, mask_tensor):
    filter_tensor = 1 - mask_tensor
    grad_pass  = in_tensor.mul(filter_tensor)
    grad_block = in_tensor.detach().mul(mask_tensor)
    return grad_pass + grad_block
    
class FastSeqProp(nn.Module):
    """
    Fast SeqProp module for sequence optimization.

    This class implements the sequence optimization algorithm Fast SeqProp

    Methods:
        add_generator_specific_args(parent_parser): Static method to add generator-specific arguments to a parser.
        process_args(grouped_args): Static method to process grouped arguments.
        __init__(energy_fn, params): Initialize the FastSeqProp optimizer.
        run(n_steps, learning_rate, step_print, lr_scheduler, create_plot, log_param_hist): Run the optimization process.
        generate(n_proposals, energy_threshold, max_attempts, n_steps, learning_rate, step_print,
                 lr_scheduler, create_plot): Generate optimized sequences.

    Note:
        - This class is designed for sequence optimization using the FastSeqProp algorithm.

    """
    
    @staticmethod
    def add_generator_specific_args(parent_parser):
        """
        Static method to add generator-specific arguments to a parser.

        Args:
            parent_parser (ArgumentParser): Parent argument parser.

        Returns:
            ArgumentParser: Argument parser with added generator-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Generator Constructor args')
        # Empty
        
        group  = parser.add_argument_group('Generator Runtime args')
        group.add_argument('--n_steps', type=int, default=20)
        group.add_argument('--learning_rate', type=float, default=0.5)
        group.add_argument('--step_print', type=int, default=10)
        group.add_argument('--lr_scheduler', type=utils.str2bool, default=True)

        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Static method to process grouped arguments.

        Args:
            grouped_args (dict): Dictionary containing grouped arguments.

        Returns:
            tuple: A tuple containing constructor args and runtime args.
        """
        constructor_args = grouped_args['Generator Constructor args']
        runtime_args     = grouped_args['Generator Runtime args']
        
        return constructor_args, runtime_args

    def __init__(self,
                 energy_fn,
                 params
                ):
        """
        Initialize the FastSeqProp optimizer.

        Args:
            energy_fn (callable): A function to evaluate the energy of sequences.
            params (object): Object containing sequence parameters.
        """
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params                           

        try: self.energy_fn.eval()
        except: pass
    
    def run(self, n_steps=20, learning_rate=0.5, step_print=10, lr_scheduler=True, grad_mask=None, create_plot=True, log_param_hist=False):
        """
        Run the optimization process using FastSeqProp.

        Args:
            n_steps (int): Number of optimization steps.
            learning_rate (float): Learning rate for optimization.
            step_print (int): Print status after this many steps.
            lr_scheduler (bool): Use learning rate scheduler.
            create_plot (bool): Create an energy plot.
            log_param_hist (bool): Log parameter history.

        Returns:
            None
        """
        if lr_scheduler: etaMin = 1e-6
        else: etaMin = learning_rate
        
        optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=etaMin)  
        
        energy_hist = []
        param_hist = [ ]
        pbar = tqdm(range(1, n_steps+1), desc='Steps', position=0, leave=True)
        for step in pbar:
            optimizer.zero_grad()
            sampled_nucleotides = self.params()
            if grad_mask is not None:
                sampled_nucleotides = mask_gradients(sampled_nucleotides, grad_mask)
            energy = self.energy_fn(sampled_nucleotides)
            energy = self.params.rebatch( energy )
            energy_hist.append(energy.detach().cpu().numpy())
            energy = energy.mean()
            energy.backward()
            optimizer.step()
            scheduler.step()
            if log_param_hist:
                param_hist.append(np.copy(self.params.theta.detach().cpu().numpy()))
            if step % step_print == 0:
                pbar.set_postfix({'Loss': energy.item(), 'LR': scheduler.get_last_lr()[0]})
        
        self.energy_hist = np.stack( energy_hist )
        if log_param_hist:
            self.param_hist = np.stack( param_hist )
        else:
            self.param_hist = None
        if create_plot:
            bsz = self.params.theta.shape[0]
            plot_data = pd.DataFrame({
                'step': np.repeat(np.arange(n_steps), bsz),
                'energy': np.stack(energy_hist).flatten()
            })
            fig, ax = plt.subplots()
            sns.lineplot(data=plot_data, x='step',y='energy',ax=ax)
            plt.show()
            return plot_data
            
    def generate(self, n_proposals=1, energy_threshold=float("Inf"), max_attempts=10000, grad_mask=None, 
                 n_steps=20, learning_rate=0.5, step_print=10, lr_scheduler=True, create_plot=False):
        """
        Generate optimized sequences using FastSeqProp.

        Args:
            n_proposals (int): Number of proposals to generate.
            energy_threshold (float): Energy threshold for acceptance.
            max_attempts (int): Maximum attempts to generate proposals.
            n_steps (int): Number of optimization steps for each attempt.
            learning_rate (float): Learning rate for optimization.
            step_print (int): Print status after this many steps.
            lr_scheduler (bool): Use learning rate scheduler.
            create_plot (bool): Create an energy plot.

        Returns:
            dict: Dictionary containing generated sequences, energies, and acceptance rate.
        """
        batch_size, *theta_shape = self.params.theta.shape
        
        proposals = torch.randn([0,*theta_shape])
        states    = torch.randn([0,*theta_shape])
        energies  = torch.randn([0])
        
        acceptance = torch.randn([0])
        
        attempts = 0
        
        while (proposals.shape[0] < n_proposals) and (attempts < max_attempts):
            
            attempts += 1
        
            self.run(
                n_steps=n_steps, learning_rate=learning_rate, step_print=step_print, 
                lr_scheduler=lr_scheduler, grad_mask=grad_mask, create_plot=create_plot
            )
            
            with torch.no_grad():
                final_states   = self.params.theta
                try:
                    final_samples = self.params.get_sample()
                    final_energies = self.energy_fn.energy_calc( 
                        self.params.add_flanks(final_samples).flatten(0,1)
                    )
                except AttributeError:
                    final_samples  = final_states.detach().clone()
                    final_energies = self.energy_fn.energy_calc( self.params() )
                
                state_bs, energy_bs = final_states.shape[0], final_energies.shape[0]
                
                if state_bs != energy_bs:
                    rebatch_energies= final_energies.unflatten(
                        0, (energy_bs//state_bs, state_bs)
                    )
                    
                    best_sample_idx = rebatch_energies.argmin(dim=0)
                    range_slicer    = torch.arange(rebatch_energies.shape[1])

                    final_samples   = final_samples[best_sample_idx, range_slicer] \
                                        .squeeze()
                    
                    final_energies  = rebatch_energies[best_sample_idx, range_slicer] \
                                        .squeeze()
                    
                else:
                    final_samples = final_samples.squeeze()
                    
                #final_energies = self.params.rebatch( final_energies )

                energy_filter = final_energies <= energy_threshold
                    
                final_states  = final_states.detach().clone()
                final_samples = final_samples.detach().clone()
                final_energies= final_energies.detach().clone()
                
            states    = torch.cat([states,     final_states[energy_filter].cpu()], dim=0)
            proposals = torch.cat([proposals, final_samples[energy_filter].cpu()], dim=0)
            energies  = torch.cat([energies, final_energies[energy_filter].cpu()], dim=0)
            
            acceptance = torch.cat([acceptance, energy_filter.cpu().float()], dim=0)
            
            try:
                self.params.reset()
            except NotImplementedError:
                pass
            
        results = {
            'states': states[:n_proposals],
            'proposals': proposals[:n_proposals],
            'energies': energies[:n_proposals],
            'acceptance_rate': acceptance.mean()
        }
        
        return results