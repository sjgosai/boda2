import argparse
import sys
import math

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as ptl

from ..common import utils 
from .basset import get_padding, Conv1dNorm, LinearNorm

##################
# Loss functions #
##################

class MSEKLmixed(nn.Module):
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        
        preds_log_prob  = preds   - preds.exp().sum(dim=1,keepdim=True).log()
        target_log_prob = targets - targets.exp().sum(dim=1,keepdim=True).log()
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        KL_loss.mul(self.kl_scale)
        
        return combined_loss.div(self.mse_scale+self.kl_scale)

class L1KLmixed(nn.Module):
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        
        preds_log_prob  = preds   - preds.exp().sum(dim=1,keepdim=True).log()
        target_log_prob = targets - targets.exp().sum(dim=1,keepdim=True).log()
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        KL_loss.mul(self.kl_scale)
        
        return combined_loss#.div(self.mse_scale+self.kl_scale)

class MSEwithEntropy(nn.Module):
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        
    def forward(self, preds, targets):
        
        pred_entropy = nn.Softmax(dim=1)(preds)
        pred_entropy = torch.sum(- pred_entropy * torch.log(pred_entropy), dim=1)
        
        targ_entropy = nn.Softmax(dim=1)(targets)
        targ_entropy = torch.sum(- targ_entropy * torch.log(targ_entropy), dim=1)
        
        MSE_loss = self.MSE(preds, targets)
        SEE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        SEE_loss.mul(self.kl_scale)
        
        return combined_loss.div(self.mse_scale+self.kl_scale)

class L1withEntropy(nn.Module):
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        
    def forward(self, preds, targets):
        
        pred_entropy = nn.Softmax(dim=1)(preds)
        pred_entropy = torch.sum(- pred_entropy * torch.log(pred_entropy), dim=1)
        
        targ_entropy = nn.Softmax(dim=1)(targets)
        targ_entropy = torch.sum(- targ_entropy * torch.log(targ_entropy), dim=1)
        
        MSE_loss = self.MSE(preds, targets)
        SEE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        SEE_loss.mul(self.kl_scale)
        
        return combined_loss#.div(self.mse_scale+self.kl_scale)
    
##################
#     Layers     #
##################

'''
class GroupedLinear(nn.Module):
    def __init__(self, in_features, group_features, groups=1, bias=True):
        super().__init__()
        
        assert in_features % groups == 0, f"Input feature count {in_features} must be divisible by groups {groups}."
        assert out_features % groups == 0, f"Output feature count {out_features} must be divisible by groups {groups}."
        
        in_sub = in_features//groups
        out_sub= out_features//groups
        
        self.branches = OrderedDict([(f'group_{i}', nn.Linear(in_sub, group_features, bias))])
        self.groups   = groups
        self.channels = in_sub
        
    def forward(self, x):
        
        return torch.cat(
            [ self.branches[f'group_{i}'](x[:,i*self.channels:(i+1)*self.channels])  for i in range(self.groups) ]
            , dim=-1
                 )
'''

class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()
        
        self.in_group_size = in_group_size
        self.out_group_size= out_group_size
        self.groups        = groups
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias   = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        
        reorg = x.permute(1,0).reshape(self.groups, self.in_group_size, -1).permute(0,2,1)
        hook  = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0,2,1).reshape(self.out_group_size*self.groups,-1).permute(1,0)
        
        return reorg

class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x.repeat(*self.args)
    
class BranchedLinear(nn.Module):
    def __init__(self, in_features, hidden_group_size, out_group_size, 
                 n_branches=1, n_layers=1, 
                 activation='ReLU', dropout_p=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers   = n_layers
        
        self.branches = OrderedDict()
        
        self.nonlin  = getattr(nn, activation)()                               
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        
        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size
            
    def forward(self, x):
        
        hook = self.intake(x)
        
        for i in range(self.n_layers):
            hook = getattr(self, f'branched_layer_{i+1}')(hook)
            hook = self.dropout( self.nonlin(hook) )
        #hook = getattr(self, f'branched_layer_{i+2}')(hook)
            
        return hook
    
##################
#     Models     #
##################

class BassetEntropyVL(ptl.LightningModule):
    r"""Write docstring here.
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Model Module args')
        
        group.add_argument('--conv1_channels', type=int, default=300)
        group.add_argument('--conv1_kernel_size', type=int, default=19)
        
        group.add_argument('--conv2_channels', type=int, default=200)
        group.add_argument('--conv2_kernel_size', type=int, default=11)
        
        group.add_argument('--conv3_channels', type=int, default=200)
        group.add_argument('--conv3_kernel_size', type=int, default=7)
        
        group.add_argument('--n_linear_layers', type=int, default=2)
        group.add_argument('--linear_channels', type=int, default=1000)
        group.add_argument('--n_outputs', type=int, default=280)
        
        group.add_argument('--dropout_p', type=float, default=0.3)
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True)
        group.add_argument('--use_weight_norm',type=utils.str2bool, default=False)
        
        group.add_argument('--criterion_reduction', type=str, default='mean')
        group.add_argument('--mse_scale', type=float, default=1.0)
        group.add_argument('--kl_scale', type=float, default=1.0)
                
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        model_args   = grouped_args['Model Module args']
        return model_args

    def __init__(self, conv1_channels=300, conv1_kernel_size=19, 
                 conv2_channels=200, conv2_kernel_size=11, 
                 conv3_channels=200, conv3_kernel_size=7, 
                 n_linear_layers=2, linear_channels=1000, 
                 n_outputs=280, activation='ReLU', 
                 dropout_p=0.3, use_batch_norm=True, use_weight_norm=False,
                 criterion_reduction='mean', mse_scale=1.0, kl_scale=1.0):                                                
        super().__init__()        
        
        self.conv1_channels    = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        
        self.conv2_channels    = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        
        self.conv3_channels    = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        
        self.n_linear_layers   = n_linear_layers
        self.linear_channels   = linear_channels
        self.n_outputs         = n_outputs
        
        self.activation        = activation
        
        self.dropout_p         = dropout_p
        self.use_batch_norm    = use_batch_norm
        self.use_weight_norm   = use_weight_norm
        
        self.criterion_reduction=criterion_reduction
        self.mse_scale         = mse_scale
        self.kl_scale          = kl_scale
        
        self.pad1  = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(4, 
                                self.conv1_channels, self.conv1_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad2  = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, 
                                self.conv2_channels, self.conv2_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad3  = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, 
                                self.conv3_channels, self.conv3_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        
        self.pad4 = nn.ConstantPad1d((1,1), 0.)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        
        next_in_channels = self.conv3_channels*13
        
        for i in range(self.n_linear_layers):
            
            setattr(self, f'linear{i+1}', 
                    LinearNorm(next_in_channels, self.linear_channels, 
                               bias=True, 
                               batch_norm=self.use_batch_norm, 
                               weight_norm=self.use_weight_norm)
                   )
            next_in_channels = self.linear_channels

        self.output  = nn.Linear(next_in_channels, self.n_outputs)
        
        self.nonlin  = getattr(nn, self.activation)()                               
        
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.criterion = MSEKLmixed(reduction=self.criterion_reduction,
                                    mse_scale=self.mse_scale,
                                    kl_scale =self.kl_scale)
        
    def encode(self, x):
        hook = self.nonlin( self.conv1( self.pad1( x ) ) )
        hook = self.maxpool_3( hook )
        hook = self.nonlin( self.conv2( self.pad2( hook ) ) )
        hook = self.maxpool_4( hook )
        hook = self.nonlin( self.conv3( self.pad3( hook ) ) )
        hook = self.maxpool_4( self.pad4( hook ) )        
        hook = torch.flatten( hook, start_dim=1 )
        return hook
    
    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout( 
                self.nonlin( 
                    getattr(self,f'linear{i+1}')(hook)
                )
            )
        return hook
    
    def classify(self, x):
        output = self.output( x )
        return output
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output

class BassetBranched(ptl.LightningModule):
    r"""Write docstring here.
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Model Module args')
        
        group.add_argument('--conv1_channels', type=int, default=300)
        group.add_argument('--conv1_kernel_size', type=int, default=19)
        
        group.add_argument('--conv2_channels', type=int, default=200)
        group.add_argument('--conv2_kernel_size', type=int, default=11)
        
        group.add_argument('--conv3_channels', type=int, default=200)
        group.add_argument('--conv3_kernel_size', type=int, default=7)
        
        group.add_argument('--n_linear_layers', type=int, default=2)
        group.add_argument('--linear_channels', type=int, default=1000)
        group.add_argument('--linear_activation',type=str, default='ReLU')
        group.add_argument('--linear_dropout_p', type=float, default=0.3)

        group.add_argument('--n_branched_layers', type=int, default=1)
        group.add_argument('--branched_channels', type=int, default=1000)
        group.add_argument('--branched_activation',type=str, default='ReLU')
        group.add_argument('--branched_dropout_p', type=float, default=0.3)

        group.add_argument('--n_outputs', type=int, default=280)
        
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True)
        group.add_argument('--use_weight_norm',type=utils.str2bool, default=False)
        
        group.add_argument('--loss_criterion', type=str, default='MSEKLmixed')
        group.add_argument('--criterion_reduction', type=str, default='mean')
        group.add_argument('--mse_scale', type=float, default=1.0)
        group.add_argument('--kl_scale', type=float, default=1.0)
                
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        model_args   = grouped_args['Model Module args']
        return model_args

    def __init__(self, conv1_channels=300, conv1_kernel_size=19, 
                 conv2_channels=200, conv2_kernel_size=11, 
                 conv3_channels=200, conv3_kernel_size=7, 
                 n_linear_layers=2, linear_channels=1000, 
                 linear_activation='ReLU', linear_dropout_p=0.3, 
                 n_branched_layers=1, branched_channels=250, 
                 branched_activation='ReLU6', branched_dropout_p=0., 
                 n_outputs=280, loss_criterion='MSEKLmixed', 
                 criterion_reduction='mean', 
                 mse_scale=1.0, kl_scale=1.0, 
                 use_batch_norm=True, use_weight_norm=False):                                                
        super().__init__()        
        
        self.conv1_channels    = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        
        self.conv2_channels    = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        
        self.conv3_channels    = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        
        self.n_linear_layers   = n_linear_layers
        self.linear_channels   = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p  = linear_dropout_p
        
        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p= branched_dropout_p
        
        self.n_outputs         = n_outputs
        
        self.loss_criterion    = loss_criterion
        self.criterion_reduction=criterion_reduction
        self.mse_scale         = mse_scale
        self.kl_scale          = kl_scale
        
        self.use_batch_norm    = use_batch_norm
        self.use_weight_norm   = use_weight_norm
        
        self.pad1  = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(4, 
                                self.conv1_channels, self.conv1_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad2  = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, 
                                self.conv2_channels, self.conv2_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad3  = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, 
                                self.conv3_channels, self.conv3_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        
        self.pad4 = nn.ConstantPad1d((1,1), 0.)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        
        next_in_channels = self.conv3_channels*13
        
        for i in range(self.n_linear_layers):
            
            setattr(self, f'linear{i+1}', 
                    LinearNorm(next_in_channels, self.linear_channels, 
                               bias=True, 
                               batch_norm=self.use_batch_norm, 
                               weight_norm=self.use_weight_norm)
                   )
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(next_in_channels, self.branched_channels, 
                                       self.branched_channels, 
                                       self.n_outputs, self.n_branched_layers, 
                                       self.branched_activation, self.branched_dropout_p)
            
        self.output  = GroupedLinear(self.branched_channels, 1, self.n_outputs)
        
        self.nonlin  = getattr(nn, self.linear_activation)()                               
        
        self.dropout = nn.Dropout(p=self.linear_dropout_p)
        
        self.criterion =  globals()[self.loss_criterion](
            reduction=self.criterion_reduction,
            mse_scale=self.mse_scale,
            kl_scale =self.kl_scale
        )
        
    def encode(self, x):
        hook = self.nonlin( self.conv1( self.pad1( x ) ) )
        hook = self.maxpool_3( hook )
        hook = self.nonlin( self.conv2( self.pad2( hook ) ) )
        hook = self.maxpool_4( hook )
        hook = self.nonlin( self.conv3( self.pad3( hook ) ) )
        hook = self.maxpool_4( self.pad4( hook ) )        
        hook = torch.flatten( hook, start_dim=1 )
        return hook
    
    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout( 
                self.nonlin( 
                    getattr(self,f'linear{i+1}')(hook)
                )
            )
        hook = self.branched(hook)

        return hook
    
    def classify(self, x):
        output = self.output( x )
        return output
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output