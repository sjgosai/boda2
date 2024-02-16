import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from ..common import constants, utils

class ZeroOrderMarkov(nn.Module):
    
    """Use with PassThroughParameters"""
    
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
        group.add_argument('--n_steps', type=int, default=1000)

        return parser
    
    def __init__(self, params, energy_fn):
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        
        try: self.energy_fn.eval()
        except: pass

    def generate(self, n_proposals, n_steps=1000, log_energy=False, log_step=1):
        with torch.no_grad():
            self.energy_log = []

            # Get shape information
            token_dim = self.params.token_dim
            batch_dim = self.params.batch_dim
            assert batch_dim == 0, "batch_dim must be 0" # eventually update for variable batch dim

            num_classes = self.params.theta.shape[token_dim]
            reorder = list(range(len(self.params.theta.shape)-1))
            reorder.insert(token_dim, -1)

            # Initialize eventual outputs
            init_proposals_shape = list(self.params.theta.shape)
            init_proposals_shape[batch_dim] = 0
            proposals = torch.zeros(
                init_proposals_shape, dtype=torch.float,
                layout=self.params.theta.layout, 
                device=self.params.theta.device
            )
            energies  = torch.zeros(
                (0,), dtype=torch.float, 
                layout=self.params.theta.layout, 
                device=self.params.theta.device
            )

            # Iterate through samples
            pbar = tqdm.tqdm(range(n_steps), desc='Steps', position=0, leave=True)
            for i in pbar:
                sample = torch.randn_like(self.params.theta)
                sample = torch.argmax( sample, dim=token_dim )
                sample = F.one_hot(sample, num_classes=num_classes).permute(*reorder).float()

                energy = self.params.rebatch( self.energy_fn( self.params(sample) ) )

                energies, indices = torch.sort(torch.cat([energies, energy]), dim=0)
                proposals = torch.cat([proposals, sample], dim=batch_dim)[indices] # eventually update for variable batch dim

                energies = energies[:n_proposals]
                proposals= proposals[:n_proposals]

                pbar.set_postfix({'energy': torch.median(energies).item()})
                if log_energy and (i % log_step == 0):
                    self.energy_log.append(
                        np.stack(
                            [np.array([i]*energies.numel()), 
                             energies.clone().detach().cpu().numpy()]
                        ).T
                    )

            results = {
                'proposals': proposals[:n_proposals],
                'energies': energies[:n_proposals],
            }
            
            if log_energy:
                self.energy_log = pd.DataFrame(data=np.concatenate(self.energy_log), columns=['step','energy'])
        return results