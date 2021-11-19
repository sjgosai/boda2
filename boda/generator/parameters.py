import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
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
    
    def prior_nll(self):
        raise NotImplementedError("Prior Negative Log-Likelihood not implemented.")
        return None
    
class BasicParameters(ParamsBase):
    
    @staticmethod
    def add_param_specific_args(parent_parser):
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Param Module args')
        
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
        
        param_args = grouped_args['Param Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    param_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(param_args.batch_size,
                                 param_args.n_channels,
                                 param_args.length)
            param_args.data = dist.OneHotCategorical(logits=logits.permute(0,2,1)) \
                     .sample().permute(0,2,1)
        
        param_args.left_flank = utils.dna2tensor( 
            param_args.left_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)

        param_args.right_flank= utils.dna2tensor( 
            param_args.right_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)
        
        del param_args.batch_size
        del param_args.n_channels
        del param_args.length
        del params_args.init_seqs
        
        return param_args
        
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
        my_attr = [ getattr(self, x) for x in ['left_flank', 'theta', 'right_flank'] ]
        return torch.cat( [ x for x in my_attr if x is not None ], axis=self.cat_axis )
    
    def rebatch(self, input):
        return input

class StraightThroughParameters(ParamsBase):

        @staticmethod
    def add_param_specific_args(parent_parser):
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Param Module args')
        
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
        
        param_args = grouped_args['Param Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    param_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(param_args.batch_size,
                                 param_args.n_channels,
                                 param_args.length)
            param_args.data = dist.OneHotCategorical(logits=logits.permute(0,2,1)) \
                     .sample().permute(0,2,1)
        
        param_args.left_flank = utils.dna2tensor( 
            param_args.left_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)

        param_args.right_flank= utils.dna2tensor( 
            param_args.right_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)
        
        del param_args.batch_size
        del param_args.n_channels
        del param_args.length
        del params_args.init_seqs
        
        return param_args
        
    def __init__(self,
                 data, 
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 token_dim=-2,
                 cat_axis=-1,
                 n_samples=1,
                 use_affine=True,
                 **kwrags):
        super().__init__()

        self.theta = nn.Parameter(data)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)
        
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.n_samples = n_samples
        self.use_affine = use_affine
        
        self.num_classes= self.theta.shape[self.token_dim]
        self.n_dims     = len(self.theta.shape)
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
        sampled_idxs = Categorical( torch.transpose(probs, self.token_dim, self.cat_axis) )
        samples = sampled_idxs.sample( (self.n_samples, ) )
        samples = F.one_hot(samples, num_classes=self.num_classes)
        samples = torch.transpose(samples, self.token_dim, self.cat_axis)
        probs = probs.repeat( self.n_samples, *[1 for i in range(self.n_dims)] )
        samples = samples - probs.detach() + probs
        return samples
        
    def forward(self):
        pieces = []    
        if self.left_flank is not None:
            pieces.append(self.left_flank.repeat(self.n_samples, self.batch_size, *[1 for i in range(self.n_dims-1)]))          
        pieces.append(self.get_sample())  
        if self.right_flank is not None:
            pieces.append(self.right_flank.repeat(self.n_samples, self.batch_size, *[1 for i in range(self.n_dims-1)]))          
        return torch.cat(pieces, axis=self.cat_axis ).flatten(0,1)
        
    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

class GumbelSoftmaxParameters(ParamsBase):
    
    @staticmethod
    def add_param_specific_args(parent_parser):
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Param Module args')
        
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
        
        param_args = grouped_args['Param Module args']
        
        if params_args.init_seqs is not None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if 'gs://' in params_args.init_seqs:
                    subprocess.call(['gsutil','cp',params_args.init_seqs,tmpdirname])
                    filename = os.path.basename(params_args.init_seqs)
                    params_args.init_seqs = os.path.join([tmpdirname, filename])
                    
                with open(params_args.init_seqs, 'r') as f:
                    param_args.data = torch.stack(
                        [ utils.dna2tensor(line) for line in f.readlines() ], 
                        dim=0
                    )
        
        else:
            logits = torch.randn(param_args.batch_size,
                                 param_args.n_channels,
                                 param_args.length)
            param_args.data = dist.OneHotCategorical(logits=logits.permute(0,2,1)) \
                     .sample().permute(0,2,1)
        
        param_args.left_flank = utils.dna2tensor( 
            param_args.left_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)

        param_args.right_flank= utils.dna2tensor( 
            param_args.right_flank 
        ).unsqueeze(0).expand(param_args.data.shape[0], -1, -1)
        
        del param_args.batch_size
        del param_args.n_channels
        del param_args.length
        del params_args.init_seqs
        
        return param_args
        
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
        self.register_parameter('theta', data)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)
        
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
    
    def forward(self):
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.repeat(self.n_samples, *self.repeater) )
            
        pieces.append( self.get_sample() )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.repeat(self.n_samples, *self.repeater) )
            
        return torch.cat( pieces, axis=self.cat_axis ).flatten(0,1)
        
    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)

#     def prior_nll(self):
#         return self.theta.transpose(self.batch_dim, 0).flatten(1).pow(2).div(2*self.prior_var).mean(1)