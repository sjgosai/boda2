import sys
import time
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torch.distributions as dist
from torch.distributions.categorical import Categorical

from ..common import constants, utils

class ParamsBase(nn.Module):
    """
    Base class for representing sequence parameters.

    Methods:
        shape: Get the shape of the parameters.
        forward: Compute the forward pass for the parameters.
        rebatch(input): Rebatch the parameters based on input.
        reset(input): Reset the parameters based on input.
        prior_nll(): Compute the Negative Log-Likelihood of the prior for the parameters.
    """
    
    def __init__(self):
        super().__init__()

    @property
    def shape(self):
        """
        Get the shape of the parameters.

        Returns:
            tuple: Shape of the parameters.
        """
        return self().shape
    
    def forward(self):
        """
        Compute the forward pass for the parameters.

        Raises:
            NotImplementedError: If not implemented by derived class.

        Returns:
            None
        """
        raise NotImplementedError("Params forward call not implemented.")
        return None
        
    def rebatch(self, input):
        """
        Rebatch the parameters based on input.

        Args:
            input: Input data for rebatching.

        Raises:
            NotImplementedError: If not implemented by derived class.

        Returns:
            None
        """
        raise NotImplementedError("Rebatch function not implemented.")
        return None
    
    def reset(self, input):
        """
        Reset the parameters based on input.

        Args:
            input: Input data for resetting parameters.

        Raises:
            NotImplementedError: If not implemented by derived class.

        Returns:
            None
        """
        raise NotImplementedError("Reset function not implemented.")
        return None
    
    def prior_nll(self):
        """
        Compute the Negative Log-Likelihood of the prior for the parameters.

        Raises:
            NotImplementedError: If not implemented by derived class.

        Returns:
            None
        """
        raise NotImplementedError("Prior Negative Log-Likelihood not implemented.")
        return None
    
class PassThroughParameters(ParamsBase):
    """
    Class representing pass-through sequence parameters.
    
    Args:
        data (torch.Tensor): Tensor containing sequence data.
        left_flank (torch.Tensor, optional): Tensor containing left flanking sequence data. Defaults to None.
        right_flank (torch.Tensor, optional): Tensor containing right flanking sequence data. Defaults to None.
        batch_dim (int, optional): Batch dimension for the data tensor. Defaults to 0.
        cat_axis (int, optional): Axis along which to concatenate the flanks and sample. Defaults to -1.
    
    Attributes:
        theta (torch.Tensor): Sequence data tensor.
        left_flank (torch.Tensor): Left flanking sequence data tensor.
        right_flank (torch.Tensor): Right flanking sequence data tensor.
        batch_dim (int): Batch dimension for the data tensor.
        cat_axis (int): Axis along which to concatenate the flanks and sample.

    Methods:
        add_flanks(my_sample): Adds flanking sequences to a sample.
        forward(my_sample): Adds flanking sequences to the input sample.
    """
    
    @staticmethod
    def add_params_specific_args(parent_parser):
        """
        Add command-line arguments specific to sequence parameters.
        
        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.
            
        Returns:
            argparse.ArgumentParser: Argument parser with added sequence parameter arguments.
        """

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Params Module args')
        
        group.add_argument('--batch_size', type=int, default=1)
        group.add_argument('--n_channels', type=int, default=4)
        group.add_argument('--length', type=int, default=200)
        group.add_argument('--init_seqs', type=str)
        group.add_argument('--left_flank', type=str, 
                           default=constants.MPRA_UPSTREAM[-200:])
        group.add_argument('--right_flank', type=str, 
                           default=constants.MPRA_DOWNSTREAM[:200])
        group.add_argument('--batch_dim', type=int, default=0)
        group.add_argument('--cat_axis', type=int, default=-1)
        
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        """
        Process arguments related to sequence parameters.
        
        Args:
            grouped_args (Namespace): Namespace containing grouped arguments.
            
        Returns:
            Namespace: Processed arguments related to sequence parameters.
        """
        params_args = grouped_args['Params Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    params_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(params_args.batch_size,
                                 params_args.n_channels,
                                 params_args.length)
            params_args.data = logits
        
        if params_args.left_flank is not None:
            params_args.left_flank = utils.dna2tensor( 
                params_args.left_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)

        if params_args.right_flank is not None:
            params_args.right_flank= utils.dna2tensor( 
                params_args.right_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)
        
        del params_args.batch_size
        del params_args.n_channels
        del params_args.length
        del params_args.init_seqs
        
        return params_args
        
    def __init__(self,
                 data,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1
                ):
        """
        Initialize the PassThroughParameters object.
        
        Args:
            data (torch.Tensor): Tensor containing sequence data.
            left_flank (torch.Tensor, optional): Tensor containing left flanking sequence data. Defaults to None.
            right_flank (torch.Tensor, optional): Tensor containing right flanking sequence data. Defaults to None.
            batch_dim (int, optional): Batch dimension for the data tensor. Defaults to 0.
            cat_axis (int, optional): Axis along which to concatenate the flanks and sample. Defaults to -1.
        """
        super().__init__()
        
        self.register_buffer('theta', data.detach().clone())
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.batch_dim = batch_dim
        self.cat_axis  = cat_axis
        
    def add_flanks(self, my_sample):
        """
        Adds flanking sequences to a sample.
        
        Args:
            my_sample (torch.Tensor): Input sample tensor.
            
        Returns:
            torch.Tensor: Sample tensor with flanking sequences added.
        """
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self, my_sample):
        """
        Adds flanking sequences to the input sample.
        
        Args:
            my_sample (torch.Tensor): Input sample tensor.
            
        Returns:
            torch.Tensor: Sample tensor with flanking sequences added.
        """
        return self.add_flanks(my_sample)
    
class BasicParameters(ParamsBase):
    """
    Class representing basic sequence parameters.
    
    Args:
        data (torch.Tensor): Tensor containing sequence data.
        left_flank (torch.Tensor, optional): Tensor containing left flanking sequence data. Defaults to None.
        right_flank (torch.Tensor, optional): Tensor containing right flanking sequence data. Defaults to None.
        batch_dim (int, optional): Batch dimension for the data tensor. Defaults to 0.
        token_dim (int, optional): Token dimension for the data tensor. Defaults to -2.
        cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Defaults to -1.
    
    Attributes:
        theta (nn.Parameter): Sequence data parameter.
        left_flank (torch.Tensor): Left flanking sequence data tensor.
        right_flank (torch.Tensor): Right flanking sequence data tensor.
        batch_dim (int): Batch dimension for the data tensor.
        token_dim (int): Token dimension for the data tensor.
        cat_axis (int): Axis along which to concatenate the flanks and the sample.

    Methods:
        add_flanks(x): Adds flanking sequences to a tensor.
        forward(x): Adds flanking sequences to the input tensor.
        reset(): Resets the parameter values to random samples.
        rebatch(input): Rebatching function.
    """
    
    @staticmethod
    def add_params_specific_args(parent_parser):
        """
        Add command-line arguments specific to basic sequence parameters.
        
        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.
            
        Returns:
            argparse.ArgumentParser: Argument parser with added basic sequence parameter arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Params Module args')
        
        group.add_argument('--batch_size', type=int, default=1)
        group.add_argument('--n_channels', type=int, default=4)
        group.add_argument('--length', type=int, default=200)
        group.add_argument('--init_seqs', type=str)
        group.add_argument('--left_flank', type=str, 
                           default=constants.MPRA_UPSTREAM[-200:])
        group.add_argument('--right_flank', type=str, 
                           default=constants.MPRA_DOWNSTREAM[:200])
        group.add_argument('--batch_dim', type=int, default=0)
        group.add_argument('--token_dim', type=int, default=-2)
        group.add_argument('--cat_axis', type=int, default=-1)
        
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        """
        Process arguments related to basic sequence parameters.
        
        Args:
            grouped_args (Namespace): Namespace containing grouped arguments.
            
        Returns:
            Namespace: Processed arguments related to basic sequence parameters.
        """
        params_args = grouped_args['Params Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    params_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(params_args.batch_size,
                                 params_args.n_channels,
                                 params_args.length)
            params_args.data = dist.OneHotCategorical(logits=logits.permute(0,2,1)) \
                     .sample().permute(0,2,1)
        
        if params_args.left_flank is not None:
            params_args.left_flank = utils.dna2tensor( 
                params_args.left_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)

        if params_args.right_flank is not None:
            params_args.right_flank= utils.dna2tensor( 
                params_args.right_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)
        
        del params_args.batch_size
        del params_args.n_channels
        del params_args.length
        del params_args.init_seqs
        
        return params_args
        
    def __init__(self,
                 data,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 token_dim=-2,
                 cat_axis=-1
                ):
        """
        Initialize the BasicParameters object.
        
        Args:
            data (torch.Tensor): Tensor containing sequence data.
            left_flank (torch.Tensor, optional): Tensor containing left flanking sequence data. Defaults to None.
            right_flank (torch.Tensor, optional): Tensor containing right flanking sequence data. Defaults to None.
            batch_dim (int, optional): Batch dimension for the data tensor. Defaults to 0.
            token_dim (int, optional): Token dimension for the data tensor. Defaults to -2.
            cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Defaults to -1.
        """
        super().__init__()
        
        self.register_parameter('theta', nn.Parameter(data.detach().clone()))
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.cat_axis  = cat_axis
        
    @property
    def shape(self):
        """
        Get the shape of the parameter tensor.
        
        Returns:
            torch.Size: Shape of the parameter tensor.
        """
        return self().shape

    def add_flanks(self, x):
        """
        Adds flanking sequences to a tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Tensor with flanking sequences added.
        """
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank )
            
        pieces.append( x )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank )
            
        return torch.cat( pieces, axis=self.cat_axis )

    def forward(self, x=None):
        """
        Adds flanking sequences to the input tensor.
        
        Args:
            x (torch.Tensor, optional): Input tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Tensor with flanking sequences added.
        """
        if x is None:
            x = self.theta
        return self.add_flanks(x)
    
    def reset(self):
        """
        Resets the parameter values to random samples from a categorical distribution.
        """
        logits = torch.randn_like( self.theta )
        theta_n= dist.OneHotCategorical(logits=logits.transpose(-1,self.token_dim)) \
                   .sample().transpose(-1,self.token_dim)
        self.theta.data = theta_n
        return None
    
    def rebatch(self, input):
        """
        Rebatching function.
        
        Args:
            input: Input data.
            
        Returns:
            input: Rebatched data.
        """
        return input

class StraightThroughParameters(ParamsBase):
    """
    Parameters class that implements the Straight-Through estimator.
    
    Args:
        data (torch.Tensor): The initial parameter tensor.
        left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
        right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
        batch_dim (int, optional): Batch dimension. Default is 0.
        token_dim (int, optional): Token dimension. Default is -2.
        cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Default is -1.
        n_samples (int, optional): Number of samples to draw. Default is 1.
        use_affine (bool, optional): Whether to use affine transformation in InstanceNorm. Default is True.
    """

    @staticmethod
    def add_params_specific_args(parent_parser):
        """
        Add command-line arguments specific to the StraightThroughParameters module.
        
        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.
        
        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Params Module args')
        
        group.add_argument('--batch_size', type=int, default=1)
        group.add_argument('--n_channels', type=int, default=4)
        group.add_argument('--length', type=int, default=200)
        group.add_argument('--init_seqs', type=str)
        group.add_argument('--left_flank', type=str, 
                           default=constants.MPRA_UPSTREAM[-200:])
        group.add_argument('--right_flank', type=str, 
                           default=constants.MPRA_DOWNSTREAM[:200])
        group.add_argument('--batch_dim', type=int, default=0)
        group.add_argument('--token_dim', type=int, default=-2)
        group.add_argument('--cat_axis', type=int, default=-1)
        group.add_argument('--n_samples', type=int, default=1)
        group.add_argument('--use_affine', type=utils.str2bool, default=False)
        
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        """
        Process the command-line arguments for the StraightThroughParameters module.
        
        Args:
            grouped_args (dict): Grouped command-line arguments.
        
        Returns:
            Namespace: Processed arguments.
        """
        params_args = grouped_args['Params Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    params_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(params_args.batch_size,
                                 params_args.n_channels,
                                 params_args.length)
            params_args.data = logits
        
        if params_args.left_flank is not None:
            params_args.left_flank = utils.dna2tensor( 
                params_args.left_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)

        if params_args.right_flank is not None:
            params_args.right_flank= utils.dna2tensor( 
                params_args.right_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)
        
        del params_args.batch_size
        del params_args.n_channels
        del params_args.length
        del params_args.init_seqs
        
        return params_args
        
    def __init__(self,
                 data, 
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 token_dim=-2,
                 cat_axis=-1,
                 n_samples=1,
                 use_affine=True):
        """
        Initialize StraightThroughParameters.
        
        Args:
            data (torch.Tensor): The initial parameter tensor.
            left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
            right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
            batch_dim (int, optional): Batch dimension. Default is 0.
            token_dim (int, optional): Token dimension. Default is -2.
            cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Default is -1.
            n_samples (int, optional): Number of samples to draw. Default is 1.
            use_affine (bool, optional): Whether to use affine transformation in InstanceNorm. Default is True.
        """
        super().__init__()

        self.register_parameter('theta', nn.Parameter(data.detach().clone()))
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.n_samples = n_samples
        self.use_affine = use_affine
        
        self.num_classes= self.theta.shape[self.token_dim]
        self.n_dims     = len(self.theta.shape)
        self.repeater   = [ 1 for i in range(self.n_dims) ]
        self.batch_size = self.theta.shape[self.batch_dim]

        self.instance_norm = nn.InstanceNorm1d(num_features=self.num_classes, affine=self.use_affine)
        
    @property
    def shape(self):
        """
        Get the shape of the parameter tensor.
        
        Returns:
            torch.Size: Shape of the parameter tensor.
        """
        return self.get_logits().shape
        
    def get_logits(self):
        """
        Get the concatenated logits from the parameter components.

        Returns:
            torch.Tensor: Concatenated logits.
        """
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
        
    def get_probs(self, x=None):
        """
        Get the probabilities using Gumbel-Softmax estimator.

        Args:
            x (torch.Tensor, optional): Input tensor. Default is None.

        Returns:
            torch.Tensor: Probabilities tensor.
        """
        if x is None:
            x = self.theta
        logits = self.instance_norm(x)
        return F.softmax(logits, dim=self.token_dim)
        
    def get_sample(self, x=None):
        """
        Get the sampled tensor using Straight-Through Gumbel-Softmax estimator.

        Args:
            x (torch.Tensor, optional): Input tensor. Default is None.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        probs = self.get_probs(x)
        probs_t = torch.transpose(probs, self.token_dim, self.cat_axis)
        sampled_idxs = Categorical( probs_t )
        samples = sampled_idxs.sample( (self.n_samples, ) )
        samples = F.one_hot(samples, num_classes=self.num_classes)
        samples = torch.transpose(samples, self.token_dim, self.cat_axis)
        probs = probs.repeat( self.n_samples, *[1 for i in range(self.n_dims)] )
        samples = samples - probs.detach() + probs
        return samples
    
    def add_flanks(self, my_sample):
        """
        Add flanks to the given sample.

        Args:
            my_sample (torch.Tensor): Input sample tensor.

        Returns:
            torch.Tensor: Tensor with flanks added.
        """
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.repeat(self.n_samples, *self.repeater) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.repeat(self.n_samples, *self.repeater) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self, x=None):
        """
        Forward pass through the StraightThroughParameters module.

        Args:
            x (torch.Tensor, optional): Input tensor. Default is None.

        Returns:
            torch.Tensor: Forward pass result.
        """
        return self.add_flanks( self.get_sample(x) ).flatten(0,1)
                
    def reset(self):
        """
        Reset the parameter tensor to random values.
        """
        self.theta.data = torch.randn_like( self.theta )
        return None
        
    def rebatch(self, input):
        """
        Rebatch the input tensor to the original shape.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rebatched tensor.
        """
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

class GumbelSoftmaxParameters(ParamsBase):
    """
    Parameters class that implements Gumbel-Softmax relaxation.

    Args:
        data (torch.Tensor): The initial parameter tensor.
        left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
        right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
        batch_dim (int, optional): Batch dimension. Default is 0.
        token_dim (int, optional): Token dimension. Default is 1.
        cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Default is -1.
        n_samples (int, optional): Number of samples to draw. Default is 1.
        tau (float, optional): Temperature parameter for Gumbel-Softmax. Default is 1.0.
        prior_var (float, optional): Variance of the prior distribution. Default is 1.0.
        use_norm (bool, optional): Whether to use InstanceNorm. Default is False.
        use_affine (bool, optional): Whether to use affine transformation in InstanceNorm. Default is False.
    """
    
    @staticmethod
    def add_params_specific_args(parent_parser):
        """
        Add command-line arguments specific to the GumbelSoftmaxParameters module.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Params Module args')
        
        group.add_argument('--batch_size', type=int, default=1)
        group.add_argument('--n_channels', type=int, default=4)
        group.add_argument('--length', type=int, default=200)
        group.add_argument('--init_seqs', type=str)
        group.add_argument('--left_flank', type=str, 
                           default=constants.MPRA_UPSTREAM[-200:])
        group.add_argument('--right_flank', type=str, 
                           default=constants.MPRA_DOWNSTREAM[:200])
        group.add_argument('--batch_dim', type=int, default=0)
        group.add_argument('--token_dim', type=int, default=1)
        group.add_argument('--cat_axis', type=int, default=-1)
        group.add_argument('--n_samples', type=int, default=1)
        group.add_argument('--tau', type=float, default=1.)
        group.add_argument('--prior_var', type=float, default=1.)
        group.add_argument('--use_norm', type=utils.str2bool, default=False)
        group.add_argument('--use_affine', type=utils.str2bool, default=False)
        
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        """
        Process the command-line arguments for the GumbelSoftmaxParameters module.

        Args:
            grouped_args (dict): Grouped command-line arguments.

        Returns:
            Namespace: Processed arguments.
        """
        params_args = grouped_args['Params Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    params_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(params_args.batch_size,
                                 params_args.n_channels,
                                 params_args.length)
            params_args.data = dist.OneHotCategorical(logits=logits.permute(0,2,1)) \
                     .sample().permute(0,2,1)
        
        if params_args.left_flank is not None:
            params_args.left_flank = utils.dna2tensor( 
                params_args.left_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)

        if params_args.right_flank is not None:
            params_args.right_flank= utils.dna2tensor( 
                params_args.right_flank 
            ).unsqueeze(0).expand(params_args.data.shape[0], -1, -1)
        
        del params_args.batch_size
        del params_args.n_channels
        del params_args.length
        del params_args.init_seqs
        
        return params_args
        
    def __init__(self,
                 data, 
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 token_dim=1,
                 cat_axis=-1,
                 n_samples=1,
                 tau=1.,
                 prior_var=1.,
                 use_norm=False,
                 use_affine=False
                ):
        """
        Initialize GumbelSoftmaxParameters.

        Args:
            data (torch.Tensor): The initial parameter tensor.
            left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
            right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
            batch_dim (int, optional): Batch dimension. Default is 0.
            token_dim (int, optional): Token dimension. Default is 1.
            cat_axis (int, optional): Axis along which to concatenate the flanks and the sample. Default is -1.
            n_samples (int, optional): Number of samples to draw. Default is 1.
            tau (float, optional): Temperature parameter for Gumbel-Softmax. Default is 1.0.
            prior_var (float, optional): Variance of the prior distribution. Default is 1.0.
            use_norm (bool, optional): Whether to use InstanceNorm. Default is False.
            use_affine (bool, optional): Whether to use affine transformation in InstanceNorm. Default is False.
        """
        super().__init__()
        self.register_parameter('theta', nn.Parameter(data.detach().clone()))
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.n_samples = n_samples
        self.tau = tau
        self.prior_var = prior_var
        
        self.use_norm = use_norm
        self.use_affine = use_affine
        
        self.num_classes= self.theta.shape[self.token_dim]
        self.n_dims     = len(self.theta.shape)
        self.repeater   = [ -1 for i in range(self.n_dims) ]
        self.batch_size = self.theta.shape[self.batch_dim]
        
        if self.use_norm:
            self.norm = nn.InstanceNorm1d(num_features=self.num_classes, 
                                          affine=self.use_affine)
        else:
            self.norm = nn.Identity()
        
    @property
    def shape(self):
        """
        Get the shape of the parameter tensor.

        Returns:
            torch.Size: Shape of the parameter tensor.
        """
        return self.get_logits().shape
    
    def get_logits(self):
        """
        Get the concatenated logits from the parameter components.

        Returns:
            torch.Tensor: Concatenated logits.
        """
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def get_sample(self, x=None):
        """
        Get the Gumbel-Softmax relaxed sample.

        Args:
            x (torch.Tensor, optional): Input tensor. Default is None.

        Returns:
            torch.Tensor: Gumbel-Softmax relaxed sample.
        """
        if x is None:
            x = self.theta
        hook = self.norm( x )
        hook = F.gumbel_softmax(
            hook.unsqueeze(0).expand(self.n_samples, *self.repeater), 
            tau=self.tau, 
            hard=True, 
            eps=1e-10, 
            dim= self.token_dim + 1 if self.token_dim >= 0 else self.token_dim
        )
        return hook
    
    def add_flanks(self, my_sample):
        """
        Add flanks to the given sample.

        Args:
            my_sample (torch.Tensor): Input sample tensor.

        Returns:
            torch.Tensor: Tensor with flanks added.
        """
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.unsqueeze(0).expand(self.n_samples, *self.repeater) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.unsqueeze(0).expand(self.n_samples, *self.repeater) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self, x=None):
        """
        Forward pass through the GumbelSoftmaxParameters module.

        Args:
            x (torch.Tensor, optional): Input tensor. Default is None.

        Returns:
            torch.Tensor: Forward pass result.
        """
        return self.add_flanks( self.get_sample(x) ).flatten(0,1)
                
    def reset(self):
        """
        Reset the parameter tensor to random values.
        """
        self.theta.data = torch.randn_like( self.theta )
        return None

    def rebatch(self, input):
        """
        Rebatch the input tensor to the original shape.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rebatched tensor.
        """
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

#     def prior_nll(self):
#         return self.theta.transpose(self.batch_dim, 0).flatten(1).pow(2).div(2*self.prior_var).mean(1)