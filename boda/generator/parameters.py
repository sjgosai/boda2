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
    def __init__(self):
        super().__init__()

    @property
    def shape(self):
        return self().shape
    
    def forward(self):
        raise NotImplementedError("Params forward call not implemented.")
        return None
        
    def rebatch(self, input):
        raise NotImplementedError("Rebatch function not implemented.")
        return None
    
    def reset(self, input):
        raise NotImplementedError("Reset function not implemented.")
        return None
    
    def prior_nll(self):
        raise NotImplementedError("Prior Negative Log-Likelihood not implemented.")
        return None
    
class BasicParameters(ParamsBase):
    
    @staticmethod
    def add_params_specific_args(parent_parser):
        
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
        
        super().__init__()
        
        self.register_parameter('theta', nn.Parameter(data.detach().clone()))
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.cat_axis  = cat_axis
        
    @property
    def shape(self):
        return self().shape

    def forward(self):
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def reset(self):
        logits = torch.randn_like( self.theta )
        theta_n= dist.OneHotCategorical(logits=logits.transpose(-1,self.token_dim)) \
                   .sample().transpose(-1,self.token_dim)
        self.theta.data = theta_n
        return None
    
    def rebatch(self, input):
        return input

class StraightThroughParameters(ParamsBase):

    @staticmethod
    def add_params_specific_args(parent_parser):
        
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
        return self.get_logits().shape
        
    def get_logits(self):
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
        
    def get_probs(self):
        logits = self.instance_norm(self.theta)
        return F.softmax(logits, dim=self.token_dim)
        
    def get_sample(self):
        probs = self.get_probs()
        probs_t = torch.transpose(probs, self.token_dim, self.cat_axis)
        sampled_idxs = Categorical( probs_t )
        samples = sampled_idxs.sample( (self.n_samples, ) )
        samples = F.one_hot(samples, num_classes=self.num_classes)
        samples = torch.transpose(samples, self.token_dim, self.cat_axis)
        probs = probs.repeat( self.n_samples, *[1 for i in range(self.n_dims)] )
        samples = samples - probs.detach() + probs
        return samples
    
    def add_flanks(self, my_sample):
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.repeat(self.n_samples, *self.repeater) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.repeat(self.n_samples, *self.repeater) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self):
        return self.add_flanks( self.get_sample() ).flatten(0,1)
                
    def reset(self):
        self.theta.data = torch.randn_like( self.theta )
        return None
        
    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

class GumbelSoftmaxParameters(ParamsBase):
    
    @staticmethod
    def add_params_specific_args(parent_parser):
        
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
        self.repeater   = [ 1 for i in range(self.n_dims) ]
        self.batch_size = self.theta.shape[self.batch_dim]
        
        if self.use_norm:
            self.norm = nn.InstanceNorm1d(num_features=self.num_classes, 
                                          affine=self.use_affine)
        else:
            self.norm = nn.Identity()
        
    @property
    def shape(self):
        return self.get_logits().shape
    
    def get_logits(self):
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def get_sample(self):
        hook = self.norm( self.theta )
        hook = [ F.gumbel_softmax(hook, tau=self.tau, hard=True, dim=1) 
                 for i in range(self.n_samples) ]
        return torch.stack(hook, dim=0)
    
    def add_flanks(self, my_sample):
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.repeat(self.n_samples, *self.repeater) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.repeat(self.n_samples, *self.repeater) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self):
        return self.add_flanks( self.get_sample() ).flatten(0,1)
                
    def reset(self):
        self.theta.data = torch.randn_like( self.theta )
        return None

    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

#     def prior_nll(self):
#         return self.theta.transpose(self.batch_dim, 0).flatten(1).pow(2).div(2*self.prior_var).mean(1)