import sys
import argparse
import inspect
import torch

from torch import nn

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
        group.add_argument('--scheduler_mode', type=str, default='min')
        group.add_argument('--factor', type=float, default=0.1)
        group.add_argument('--patience', type=int, default=10)
        group.add_argument('--threshold', type=float, default=1e-4)
        group.add_argument('--threshold_mode', type=str, default='rel')
        group.add_argument('--cooldown', type=int, default=0)
        group.add_argument('--min_lr', type=float, default=0.)
        #group.add_argument('--eps', type=float, default=1e-8)
    elif scheduler_name == 'CyclicLR':
        group.add_argument('--base_lr', type=float, required=True)
        group.add_argument('--max_lr', type=float, required=True)
        group.add_argument('--step_size_up', type=int, default=2000)
        group.add_argument('--step_size_down', type=int)
        group.add_argument('--scheduler_mode', type=str, default='triangular')
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

def reorg_scheduler_args(sched_arg_dict):
    if 'scheduler_mode' in sched_arg_dict.keys():
        sched_arg_dict['mode'] = sched_arg_dict['scheduler_mode']
        sched_arg_dict.pop('scheduler_mode')
    return sched_arg_dict

def filter_state_dict(model, stashed_dict, fill_tensor=False):
    results_dict = { 
        'filtered_state_dict': {},
        'passed_keys'  : [],
        'removed_keys' : [],
        'missing_keys' : [],
        'unloaded_keys': []
                   }
    old_dict = model.state_dict()

    for m_key, m_value in old_dict.items():
        try:
            
            if old_dict[m_key].shape == stashed_dict[m_key].shape:
                results_dict['filtered_state_dict'][m_key] = stashed_dict[m_key]
                results_dict['passed_keys'].append(m_key)
                print(f'Key {m_key} successfully matched', file=sys.stderr)
                
            else:
                check_str = 'Size mismatch for key: {}, expected size {}, got {}' \
                              .format(m_key, old_dict[m_key].shape, stashed_dict[m_key].shape)
                if fill_tensor:
                    check_str = check_str + f" \n Filling key: {m_key}"
                    weight_size = [old_dict[m_key].size(dim=0)-stashed_dict[m_key].size(dim=0), \
                                   old_dict[m_key].size(dim=1),old_dict[m_key].size(dim=2)]
                    if 'weight' in m_key:
                        extra = torch.normal(torch.zeros(weight_size), \
                                             torch.ones(weight_size)*torch.tensor(2/weight_size[1]).sqrt())
                    elif 'bias' in m_key:
                        extra = torch.normal(torch.zeros(1,1,weight_size[-1]), \
                                             torch.ones(1,1,weight_size[-1])*torch.tensor(2/weight_size[1]).sqrt())
                    results_dict['filtered_state_dict'][m_key] = torch.cat([stashed_dict[m_key], extra], dim=0)
                        
                results_dict['removed_keys'].append(m_key)
                print(check_str, file=sys.stderr)
                
        except KeyError:
            results_dict['missing_keys'].append(m_key)
            print(f'Missing key in dict: {m_key}', file=sys.stderr)
            
    for m_key, m_value in stashed_dict.items():
        if m_key not in old_dict.keys():
            check_str = 'Skipped loading key: {} of size {}' \
                           .format(m_key, m_value.shape)
            results_dict['unloaded_keys'].append(m_key)
            print(check_str, file=sys.stderr)
            
    return results_dict
                     
def pearson_correlation(x, y):
    vx = x - torch.mean(x, dim=0)
    vy = y - torch.mean(y, dim=0)
    pearsons = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)))
    return pearsons, torch.mean(pearsons)
    
def shannon_entropy(x):
    p_c = nn.Softmax(dim=1)(x)    
    return torch.sum(- p_c * torch.log(p_c), axis=1)

def _get_ranks(x):
    tmp = x.argsort(dim=0)
    ranks = torch.zeros_like(tmp)
    if len(x.shape) > 1:
        dims = x.shape[1]
        for dim in range(dims):
            ranks[tmp[:,dim], dim] = torch.arange(x.shape[0], layout=x.layout, device=x.device)
    else:
        ranks[tmp] = torch.arange(x.shape[0], layout=x.layout, device=x.device)
    return ranks

def spearman_correlation(x, y):
    x_rank = _get_ranks(x).float()
    y_rank = _get_ranks(y).float()
    return pearson_correlation(x_rank, y_rank)