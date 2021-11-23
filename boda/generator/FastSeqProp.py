import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..common import constants, utils

class FastSeqProp(nn.Module):
    @staticmethod
    def add_generator_specific_args(parent_parser):
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
        constructor_args = grouped_args['Generator Constructor args']
        runtime_args     = grouped_args['Generator Runtime args']
        
        return constructor_args, runtime_args

    def __init__(self,
                 energy_fn,
                 params
                ):
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params                           

        try: self.energy_fn.eval()
        except: pass
    
    def run(self, n_steps=20, learning_rate=0.5, step_print=10, lr_scheduler=True, create_plot=True):
     
        if lr_scheduler: etaMin = 1e-6
        else: etaMin = learning_rate
        
        optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=etaMin)  
        
        energy_hist = []
        pbar = tqdm(range(1, n_steps+1), desc='Steps', position=0, leave=True)
        for step in pbar:
            optimizer.zero_grad()
            sampled_nucleotides = self.params()
            energy = self.energy_fn(sampled_nucleotides)
            energy = self.params.rebatch( energy ).mean()
            energy.backward()
            optimizer.step()
            scheduler.step()
            energy_hist.append(energy.item())
            if step % step_print == 0:
                pbar.set_postfix({'Loss': energy.item(), 'LR': scheduler.get_last_lr()[0]})
        
        self.energy_hist = energy_hist
        if create_plot:
            plt.plot(self.energy_hist)
            plt.xlabel('Steps')
            vert_label = plt.ylabel('Energy')
            vert_label.set_rotation(90)
            plt.show()
            
    def generate(self, n_proposals=1, energy_threshold=float("Inf"), max_attempts=10000, 
                 n_steps=20, learning_rate=0.5, step_print=10, lr_scheduler=True, create_plot=False):
        
        batch_size, *theta_shape = self.params.theta.shape
        
        proposals = torch.randn([0,*theta_shape])
        states    = torch.randn([0,*theta_shape])
        energies  = torch.randn([0])
        
        attempts = 0
        
        while (proposals.shape[0] < n_proposals) and (attempts < max_attempts):
            
            attempts += 1
        
            self.run(
                n_steps=n_steps, learning_rate=learning_rate, step_print=step_print, 
                lr_scheduler=lr_scheduler, create_plot=create_plot
            )
            
            with torch.no_grad():
                final_states   = self.params.theta
                final_samples  = self.params()
                final_energies = self.energy_fn.energy_calc( final_samples )
                
                state_bs, sample_bs = final_states.shape[0], final_samples.shape[0]

                if state_bs != sample_bs:
                    rebatch_samples = final_samples.unflatten(
                        0, (sample_bs//state_bs, state_bs)
                    )
                    rebatch_energies= final_energies.unflatten(
                        0, (sample_bs//state_bs, state_bs)
                    )
                    
                    best_sample_idx = rebatch_energies.argmin(dim=0)
                    range_slicer    = torch.arange(final_energies.shape[1])

                    final_samples   =  final_samples[best_sample_idx, range_slicer]

                final_energies = self.params.rebatch( final_energies )
                                   
                energy_filter = final_energies <= energy_threshold
                
                final_states  = final_states.detach().clone()
                final_samples = final_samples.detach().clone()
                final_energies= final_energies.detach().clone()
                
            states    = torch.cat([states,     final_states[energy_filter].cpu()], dim=0)
            proposals = torch.cat([proposals, final_samples[energy_filter].cpu()], dim=0)
            energies  = torch.cat([energies, final_energies[energy_filter].cpu()], dim=0)
            
            try:
                self.params.reset()
            except NotImplementedError:
                pass
            
        results = {
            'states': states[:n_proposals],
            'proposals': proposals[:n_proposals],
            'energies': energies[:n_proposals]
        }
        
        return results