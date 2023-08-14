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
    """
    BaseEnergy class for defining energy functions in sequence optimization.

    This class serves as a base class for defining energy functions to be used in sequence optimization.
    Subclasses should implement the `energy_calc` method to compute the energy of input sequences.

    Methods:
        __init__(): Initialize the BaseEnergy class.
        forward(x_in): Compute the energy of input sequences and apply penalties if applicable.
        energy_calc(x): Calculate the energy of input sequences.

    Note:
        - Subclasses must implement the `energy_calc` method to compute energy.

    """
    
    def __init__(self):
        """
        Initialize the BaseEnergy class.

        Note:
            This constructor initializes the model attribute to None.
        """
        super().__init__()

        self.model = None
        
    def forward(self, x_in):
        """
        Compute the energy of input sequences and apply penalties if applicable.

        Args:
            x_in (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
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
        """
        Calculate the energy of input sequences.

        Args:
            x (torch.Tensor): Input sequences.

        Raises:
            NotImplementedError: Raised when the method is not implemented.

        Returns:
            torch.Tensor: Computed energy values.

        """
        raise NotImplementedError("Energy caclulation not implemented.")
        x_in = x.to(self.model.device)
        
        hook = self.model(x_in)
        # do math

        return hook
      
class OverMaxEnergy(BaseEnergy):
    """
    OverMaxEnergy class for defining energy functions based on OverMax (MinGap) values.

    This class inherits from BaseEnergy and defines an energy function that calculates
    the gap between the target (bias) cell activity and the maximum off-target (non-bias) cell
    activity of a model's output for input sequences, with an optional value-bending factor.

    Args:
        model (torch.nn.Module): The neural network model used for energy calculation.
        bias_cell (int, optional): Index of the target cell. Default is 0.
        bias_alpha (float, optional): Scaling factor for the bias term. Default is 1.0.
        bending_factor (float, optional): Bending factor applied to the model's output. Default is 0.0.
        a_min (float, optional): Minimum value allowed after bending. Default is negative infinity.
        a_max (float, optional): Maximum value allowed after bending. Default is positive infinity.

    Methods:
        add_energy_specific_args(parent_parser): Add energy-specific arguments to an argparse ArgumentParser.
        process_args(grouped_args): Process grouped arguments and return energy-related arguments.
        bend(x): Apply bending factor to the input tensor.
        energy_calc(x): Calculate the energy of input sequences based on the maximum model outputs.

    Note:
        - The `model` provided must be a neural network model compatible with PyTorch.

    """
    
    @staticmethod
    def add_energy_specific_args(parent_parser):
        """
        Add energy-specific arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
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
        """
        Process grouped arguments and return energy-related arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            dict: Processed energy-related arguments.

        """
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
        """
        Initialize the OverMaxEnergy class.

        Args:
            model (torch.nn.Module): The neural network model used for energy calculation.
            bias_cell (int, optional): Index of the target cell. Default is 0.
            bias_alpha (float, optional): Scaling factor for the bias term. Default is 1.0.
            bending_factor (float, optional): Bending factor applied to the model's output. Default is 0.0.
            a_min (float, optional): Minimum value allowed after bending. Default is negative infinity.
            a_max (float, optional): Maximum value allowed after bending. Default is positive infinity.

        """
        super().__init__()
        
        self.model = model
        self.model.eval()

        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        self.bending_factor = bending_factor
        self.a_min = a_min
        self.a_max = a_max

    def bend(self, x):
        """
        Apply bending to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with applied bending.

        """
        return x - self.bending_factor * (torch.exp(-x) - 1)
        
    def energy_calc(self, x):
        """
        Calculate the energy of input sequences based on the maximum model outputs.

        Args:
            x (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
        hook = x.to(self.model.device)
        
        hook = self.bend(self.model(hook).clamp(self.a_min,self.a_max))
        energy = hook[...,[ x for x in range(hook.shape[-1]) if x != self.bias_cell]].max(-1).values \
                 - hook[...,self.bias_cell].mul(self.bias_alpha)
        return energy

class TargetEnergy(BaseEnergy):
    """
    TargetEnergy class for defining energy functions based on target values.

    This class inherits from BaseEnergy and defines an energy function that calculates the sum of absolute
    differences between the model's output and specified target values for input sequences.

    Args:
        model (torch.nn.Module): The neural network model used for energy calculation.
        targets (list of float): List of target values to compare the model's output to.
        lambd (float, optional): Lambda value for the soft shrinkage function. Default is 0.0.
        a_min (float, optional): Minimum value allowed after clamping. Default is negative infinity.
        a_max (float, optional): Maximum value allowed after clamping. Default is positive infinity.

    Methods:
        add_energy_specific_args(parent_parser): Add energy-specific arguments to an argparse ArgumentParser.
        process_args(grouped_args): Process grouped arguments and return energy-related arguments.
        energy_calc(x): Calculate the energy of input sequences based on the sum of absolute differences.

    Note:
        - The `model` provided must be a neural network model compatible with PyTorch.
        - The energy calculation uses the soft shrinkage function and target values.

    """
    
    @staticmethod
    def add_energy_specific_args(parent_parser):
        """
        Add energy-specific arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
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
        """
        Process grouped arguments and return energy-related arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            dict: Processed energy-related arguments.

        """
        energy_args = grouped_args['Energy Module args']
        
        unpack_artifact(energy_args.model_artifact)
        model = model_fn('./artifacts')
        model.cuda()
        model.eval()
        energy_args.model = model

        del energy_args.model_artifact
        
        return energy_args

    def __init__(self, model, targets, lambd=0.0, a_min=-math.inf, a_max=math.inf):
        """
        Initialize the TargetEnergy class.

        Args:
            model (torch.nn.Module): The neural network model used for energy calculation.
            targets (list of float): List of target values to compare the model's output to.
            lambd (float, optional): Lambda value for the soft shrinkage function. Default is 0.0.
            a_min (float, optional): Minimum value allowed after clamping. Default is negative infinity.
            a_max (float, optional): Maximum value allowed after clamping. Default is positive infinity.

        """
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
        """
        Calculate the energy of input sequences based on the sum of absolute differences.

        Args:
            x (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
        hook = x.to(self.model.device)
        
        energy = self.shrink( self.model(hook).clamp(self.a_min, self.a_max) - self.targets ).abs().sum(-1)
        
        return energy

        
class EntropyEnergy(BaseEnergy):
    """
    EntropyEnergy class for defining energy functions based on the entropy of model outputs.

    This class inherits from BaseEnergy and defines an energy function that calculates the entropy of model
    outputs for input sequences. Optionally, a bias term can be added to the energy calculation.

    Args:
        model (torch.nn.Module): The neural network model used for energy calculation.
        bias_cell (int, optional): Index of the cell to apply bias to. Default is None.
        bias_alpha (float, optional): Scaling factor for the bias term. Default is 1.0.

    Methods:
        energy_calc(x): Calculate the energy of input sequences based on the entropy of model outputs.

    Note:
        - The `model` provided must be a neural network model compatible with PyTorch.
        - The energy calculation is based on the Shannon entropy of the model's output probabilities.

    """
    
    def __init__(self, model, bias_cell=None, bias_alpha=1.):
        """
        Initialize the EntropyEnergy class.

        Args:
            model (torch.nn.Module): The neural network model used for energy calculation.
            bias_cell (int, optional): Index of the cell to apply bias to. Default is None.
            bias_alpha (float, optional): Scaling factor for the bias term. Default is 1.0.

        """
        super().__init__()
        
        self.model = model
        self.model.eval()
        
        self.bias_cell = bias_cell
        self.bias_alpha= bias_alpha
        
    def energy_calc(self, x):
        """
        Calculate the energy of input sequences based on the entropy of model outputs.

        Args:
            x (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
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
    """
    BasePenalty class for defining penalty functions to be applied to input sequences.

    This class serves as a base class for defining penalty functions that can be applied to input sequences
    during energy calculations. Subclasses should implement the `penalty` method to customize the penalty
    calculation.

    Methods:
        penalty(x): Calculate the penalty for input sequences.

    Note:
        - Subclasses should override the `penalty` method to define the specific penalty calculation.

    """
    def __init__(self):
        """
        Initialize the BasePenalty class.

        """
        super().__init__()

    def penalty(self, x):
        """
        Calculate the penalty for input sequences.

        Args:
            x (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed penalty values.

        Raises:
            NotImplementedError: If the penalty method is not implemented in the subclass.

        Note:
            Subclasses should override this method to define the specific penalty calculation.

        """
        raise NotImplementedError("Penalty not implemented")
        
        hook = x
        
        return hook

def sync_width(tensor_1, tensor_2):
    """
    Synchronize the width (length) of two tensors by zero-padding the shorter tensor.

    This function pads the shorter tensor to match the width (length) of the longer tensor. It can be useful
    when working with tensors that need to have the same dimensions for certain operations.

    Args:
        tensor_1 (torch.Tensor): The first input tensor.
        tensor_2 (torch.Tensor): The second input tensor.

    Returns:
        tuple: A tuple containing the synchronized tensors. If the width of `tensor_1` is less than `tensor_2`,
        `tensor_1` is padded with zeros. If the width of `tensor_2` is less than `tensor_1`, `tensor_2` is padded
        with zeros. If both tensors have the same width, they are returned as-is.

    """
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
    """
    A class representing a penalty term based on STREME motif enrichment analysis.

    This class implements a penalty term based on motif analysis using a STREME output. It calculates
    motif scores for proposed sequences and applies a penalty based on a pool of motifs and thresholds.

    Args:
        score_pct (float): The percentage of the maximum motif score used as the threshold for penalty application.

    Attributes:
        score_pct (float): The percentage of the maximum motif score used as the threshold for penalty application.

    Methods:
        register_penalty(x): Register the penalty filters for motif analysis.
        register_threshold(x): Register the score thresholds for penalty application.
        streme_penalty(streme_output): Calculate the penalty based on STREME motif analysis.
        motif_penalty(x): Calculate the motif-based penalty for a given input.
        penalty(x): Calculate and return the penalty based on the motif analysis.
        update_penalty(proposal): Update the penalty filters and thresholds based on a new proposal.
    """

    @staticmethod
    def add_penalty_specific_args(parent_parser):
        """
        Add STREME penalty specific arguments to an argument parser.

        Args:
            parent_parser (argparse.ArgumentParser): The parent argument parser.

        Returns:
            argparse.ArgumentParser: An argument parser with added STREME penalty specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Penalty Module args')
        group.add_argument('--score_pct', type=float, default=0.3)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process STREME penalty specific arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            dict: Processed STREME penalty specific arguments.
        """
        penalty_args = grouped_args['Penalty Module args']
        return penalty_args
    
    def __init__(self, score_pct):
        """
        Initialize the STREME penalty.

        Args:
            score_pct (float): The percentage of the maximum motif score used as the threshold for penalty application.
        """
        super().__init__()
        
        self.score_pct = score_pct

    def register_penalty(self, x):
        """
        Register the penalty filters (motif PWMs).

        Args:
            x (torch.Tensor): The penalty filters to be registered.
        """
        try:
            self.penalty_filters = x.type_as(self.penalty_filters)
        except AttributeError:
            self.register_buffer('penalty_filters', x)
            
    def register_threshold(self, x):
        """
        Register the score thresholds for penalty application.

        Args:
            x (torch.Tensor): The score thresholds to be registered.
        """
        try:
            self.score_thresholds = x.type_as(self.score_thresholds)
        except AttributeError:
            self.register_buffer('score_thresholds', x)
            
    def streme_penalty(self, streme_output):
        """
        Calculate the penalty based on STREME motif enrichment analysis.

        Args:
            streme_output (dict): The output of the STREME motif analysis.
        """
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
        """
        Calculate the motif-based penalty for a given input.

        Args:
            x (torch.Tensor): The input tensor for which the penalty is calculated.

        Returns:
            torch.Tensor: The calculated motif-based penalty.
        """
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
        """
        Calculate and return the penalty based on the motif analysis.

        Args:
            x (torch.Tensor): The input tensor for which the penalty is calculated.

        Returns:
            torch.Tensor: The calculated penalty based on motif analysis.
        """
        hook = x.to(self.model.device)
        return self.motif_penalty(hook)

    def update_penalty(self, proposal):
        """
        Update the penalty filters and thresholds based on a new proposal.

        Args:
            proposal (dict): A proposal containing a new batch of sequences.

        Returns:
            dict: A summary of the update, including STREME output, filters, and score thresholds.
        """
        proposals_list = common.utils.batch2list(proposal['proposals'])
        streme_results = streme(proposals_list)
        self.streme_penalty(streme_results)
        update_summary = {
            'streme_output': streme_results,
            'filters': self.penalty_filters.detach().clone(),
            'score_thresholds': self.score_thresholds.detach().clone()
        }
        return update_summary
