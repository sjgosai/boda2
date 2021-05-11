import argparse
import inspect

from ..common import utils as cutils

def add_optimizer_specific_args(parser, optimizer_name):
    group = parser.add_argument_group('Optimizer args')
    if optimizer_name == 'Adadelta':
        group.add_argument('--lr', type=float, default=1.0)
        group.add_argument('--rho',type=float, default=0.9)
        group.add_argument('--eps',type=float, default=1e-6)
        group.add_argument('--weight_decay', type=float, default=0.)
    elif optimizer_name == 'Adagrad':
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--lr_decay',type=float, default=0.)
        group.add_argument('--weight_decay', type=float, default=0.)
        group.add_argument('--eps',type=float, default=1e-10)
    elif optimizer_name == 'Adam':
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--beta1',type=float, default=0.9)
        group.add_argument('--beta2',type=float, default=0.999)
        group.add_argument('--eps',type=float, default=1e-8)
        group.add_argument('--weight_decay', type=float, default=0.)
        group.add_argument('--amsgrad', type=cutils.str2bool, default=False)
    elif optimizer_name == 'AdamW':
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--beta1',type=float, default=0.9)
        group.add_argument('--beta2',type=float, default=0.999)
        group.add_argument('--eps',type=float, default=1e-8)
        group.add_argument('--weight_decay', type=float, default=0.)
        group.add_argument('--amsgrad', type=cutils.str2bool, default=False)
    elif optimizer_name == 'SparseAdam':
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--beta1',type=float, default=0.9)
        group.add_argument('--beta2',type=float, default=0.999)
        group.add_argument('--eps',type=float, default=1e-8)
    elif optimizer_name == 'Adamax':
        group.add_argument('--lr', type=float, default=0.002)
        group.add_argument('--beta1',type=float, default=0.9)
        group.add_argument('--beta2',type=float, default=0.999)
        group.add_argument('--eps',type=float, default=1e-8)
        group.add_argument('--weight_decay', type=float, default=0.)
    elif optimizer_name == 'ASGD':
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--lambd',type=float, default=1e-4)
        group.add_argument('--alpha',type=float, default=0.75)
        group.add_argument('--t0',type=float, default=1e6)
        group.add_argument('--weight_decay', type=float, default=0.)
    elif optimizer_name == 'RMSprop':
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--momentum',type=float, default=0.)
        group.add_argument('--alpha',type=float, default=0.99)
        group.add_argument('--eps',type=float, default=1e-8)
        group.add_argument('--centered',type=cutils.str2bool, default=False)
        group.add_argument('--weight_decay', type=float, default=0.)
    elif optimizer_name == 'Rprop':
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--eta1',type=float, default=0.5)
        group.add_argument('--eta2',type=float, default=1.2)
        group.add_argument('--step_size1',type=float, default=1e-6)
        group.add_argument('--step_size2',type=float, default=50.)
    elif optimizer_name == 'SGD':
        group.add_argument('--lr', type=float, default=0.01)
        group.add_argument('--momentum',type=float, default=0.)
        group.add_argument('--weight_decay', type=float, default=0.)
        group.add_argument('--dampening',type=float, default=0.)
        group.add_argument('--nesterov',type=cutils.str2bool, default=False)
    else:
        raise RuntimeError(f'{optimizer_name} not supported. Try: [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, RMSprop, Rprop, SGD]')
        
    return parser

def add_scheduler_specific_args(parser, scheduler_name):
    if scheduler_name is not None:
        group = parser.add_argument_group('LR Scheduler args')
    if scheduler_name == 'StepLR':
        group.add_argument('--step_size', type=int, required=True)
        group.add_argument('--gamma', type=float, default=0.1)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name == 'ExponentialLR':
        group.add_argument('--gamma', type=float, required=True)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name == 'CosineAnnealingLR':
        group.add_argument('--T_max', type=int, required=True)
        group.add_argument('--eta_min', type=float, default=0.)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name == 'ReduceLROnPlateau':
        group.add_argument('--mode', type=str, default='min')
        group.add_argument('--factor', type=float, default=0.1)
        group.add_argument('--patience', type=int, default=10)
        group.add_argument('--threshold', type=float, default=1e-4)
        group.add_argument('--threshold_mode', type=str, default='rel')
        group.add_argument('--cooldown', type=int, default=0)
        group.add_argument('--min_lr', type=float, default=0.)
        group.add_argument('--eps', type=float, default=1e-8)
    elif scheduler_name == 'CyclicLR':
        group.add_argument('--base_lr', type=float, required=True)
        group.add_argument('--max_lr', type=float, required=True)
        group.add_argument('--step_size_up', type=int, default=2000)
        group.add_argument('--step_size_down', type=int)
        group.add_argument('--mode', type=str, default='triangular')
        group.add_argument('--gamma', type=float, default=1.0)
        group.add_argument('--scale_mode', type=str, default='cycle')
        group.add_argument('--cycle_momentum', type=cutils.str2bool, default=True)
        group.add_argument('--base_momentum', type=float, default=0.8)
        group.add_argument('--max_momentum', type=float, default=0.9)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name == 'OneCycleLR':
        group.add_argument('--max_lr', type=float, required=True)
        group.add_argument('--total_steps', type=int)
        group.add_argument('--epochs', type=int)
        group.add_argument('--steps_per_epoch', type=int)
        group.add_argument('--pct_start', type=float, default=0.3)
        group.add_argument('--anneal_strategy', type=str, default='cos')
        group.add_argument('--base_momentum', type=float, default=0.85)
        group.add_argument('--max_momentum', type=float, default=0.95)
        group.add_argument('--div_factor', type=float, default=25.)
        group.add_argument('--final_div_factor', type=float, default=1e4)
        group.add_argument('--three_phase', type=cutils.str2bool, default=False)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        group.add_argument('--T_0', type=int, required=True)
        group.add_argument('--T_mult', type=int, default=1)
        group.add_argument('--eta_min', type=float, default=0.)
        group.add_argument('--last_epoch', type=int, default=-1)
    elif scheduler_name is None:
        pass
    else:
        raise RuntimeError(f'{scheduler_name} not supported. Try: [None, StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts]')
        
    return parser

def reorg_optimizer_args(optim_arg_dict):
    if 'beta1' in optim_arg_dict.keys():
        optim_arg_dict['betas'] = [optim_arg_dict['beta1'], optim_arg_dict['beta2']]
        optim_arg_dict.pop('beta1')
        optim_arg_dict.pop('beta2')
    if 'eta1' in optim_arg_dict.keys():
        optim_arg_dict['etas'] = [optim_arg_dict['eta1'], optim_arg_dict['eta2']]
        optim_arg_dict.pop('eta1')
        optim_arg_dict.pop('eta2')
    if 'step_size1' in optim_arg_dict.keys():
        optim_arg_dict['step_sizes'] = [optim_arg_dict['step_size1'], optim_arg_dict['step_size2']]
        optim_arg_dict.pop('step_size1')
        optim_arg_dict.pop('step_size2')
    return optim_arg_dict

