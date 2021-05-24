import argparse
import math
import os
import sys
import time
import tempfile
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

import hypertune

from ..common import utils
from .utils import (add_optimizer_specific_args, add_scheduler_specific_args, reorg_optimizer_args, filter_state_dict,
                    Pearson_correlation, Shannon_entropy)

class CNNBasicTraining(LightningModule):
    
    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_optimizer_specific_args(parser, known_args.optimizer)
        parser = add_scheduler_specific_args(parser, known_args.scheduler)
        return parser

    @staticmethod
    def process_args(grouped_args):
        graph_args   = grouped_args['Graph Module args']
        graph_args.optimizer_args = vars(grouped_args['Optimizer args'])
        graph_args.optimizer_args = reorg_optimizer_args(graph_args.optimizer_args)
        try:
            graph_args.scheduler_args = vars(grouped_args['LR Scheduler args'])
        except KeyError:
            graph_args.scheduler_args = None
        return graph_args

    #######################
    # Dead __init__ block #
    #######################
    
    def __init__(self, optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval= scheduler_interval
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        
    ###################
    # Non-PTL methods #
    ###################
        
    def categorical_mse(self, x, y):
        return (x - y).pow(2).mean(dim=0)
        
    def aug_log(self, internal_metrics=None, external_metrics=None):
        
        if internal_metrics is not None:
            for my_key, my_value in internal_metrics.items():
                self.log(my_key, my_value)
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=my_key,
                    metric_value=my_value,
                    global_step=self.global_step)
                
        if external_metrics is not None:
            res_str = '|'
            for my_key, my_value in external_metrics.items():
                self.log(my_key, my_value)
                res_str += ' {}: {:.5f} |'.format(my_key, my_value)
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=my_key,
                    metric_value=my_value,
                    global_step=self.global_step)
            border = '-'*len(res_str)
            print("\n".join(['',border, res_str, border,'']))
        
        return None

    #############
    # PTL hooks #
    #############
        
    def configure_optimizers(self):
        self.hpt = hypertune.HyperTune()
        params = [ x for x in self.parameters() if x.requires_grad ]
        print(f'Found {sum(p.numel() for p in params)} parameters')
        optim_class = getattr(torch.optim,self.optimizer)
        my_optimizer= optim_class(self.parameters(), **self.optimizer_args)
        if self.scheduler is not None:
            opt_dict = {
                'optimizer': my_optimizer, 
                'scheduler': getattr(torch.optim.lr_scheduler,self.scheduler)(my_optimizer, **self.scheduler_args), 
                'interval': self.scheduler_interval, 
                'name': 'learning_rate'
            }
            if self.scheduler_monitor is not None:
                opt_dict['monitor'] = self.scheduler_monitor
            return opt_dict
        else:
            return my_optimizer
    
    def training_step(self, batch, batch_idx):
        x, y   = batch
        y_hat  = self(x)
        loss   = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y   = batch
        y_hat = self(x)
        loss   = self.criterion(y_hat, y)
        self.log('valid_loss', loss)
        metric = self.categorical_mse(y_hat, y)
        return {'loss': loss, 'metric': metric, 'preds': y_hat, 'labels': y}

    def validation_epoch_end(self, val_step_outputs):       
        arit_mean = torch.stack([ batch['loss'] for batch in val_step_outputs ], dim=0) \
                      .mean()
        harm_mean = torch.stack([ batch['metric'] for batch in val_step_outputs ], dim=0) \
                      .mean(dim=0).pow(-1).mean().pow(-1)
        epoch_preds = torch.cat([batch['preds'] for batch in val_step_outputs], dim=0)
        epoch_labels  = torch.cat([batch['labels'] for batch in val_step_outputs], dim=0)
        pearsons, mean_pearson = Pearson_correlation(epoch_preds, epoch_labels)
        shannon_pred, shannon_label = Shannon_entropy(epoch_preds), Shannon_entropy(epoch_labels)
        specificity_pearson, specificity_mean_pearson = Pearson_correlation(shannon_pred, shannon_label)
        self.aug_log(external_metrics={
            'arithmetic_mean_loss': arit_mean,
            'harmonic_mean_loss': harm_mean,
            'prediction_mean_pearson': mean_pearson.item(),
            'entropy_mean_pearson': specificity_mean_pearson.item()
        })

        return None
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)       
        
class CNNTransferLearning(CNNBasicTraining):
    
    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--parent_weights', type=str, required=True)
        group.add_argument('--frozen_epochs', type=int, default=9999)
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_optimizer_specific_args(parser, known_args.optimizer)
        parser = add_scheduler_specific_args(parser, known_args.scheduler)
        return parser

    @staticmethod
    def process_args(grouped_args):
        graph_args   = grouped_args['Graph Module args']
        graph_args.optimizer_args = vars(grouped_args['Optimizer args'])
        graph_args.optimizer_args = reorg_optimizer_args(graph_args.optimizer_args)
        try:
            graph_args.scheduler_args = vars(grouped_args['LR Scheduler args'])
        except KeyError:
            graph_args.scheduler_args = None
        return graph_args

    #######################
    # Dead __init__ block #
    #######################
    
    def __init__(self, parent_weights, unfreeze_epoch=9999, 
                 optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        super().__init__(optimizer, scheduler, scheduler_monitor, 
                         scheduler_interval, optimizer_args, scheduler_args)

        self.parent_weights = parent_weights
        self.frozen_epochs  = frozen_epochs
        
    ###################
    # Non-PTL methods #
    ###################
        
    def attach_parent_weights(self, my_weights):
        parent_state_dict = torch.load(my_weights)
        if 'model_state_dict' in parent_state_dict.keys():
            parent_state_dict = parent_state_dict['model_state_dict']
            
        mod_state_dict = filter_state_dict(self, parent_state_dict)
        self.load_state_dict( mod_state_dict['filtered_state_dict'], strict=False )
        return mod_state_dict['passed_keys']
    
    #############
    # PTL hooks #
    #############
        
    def setup(self, stage='training'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            if 'gs://' in self.parent_weights:
                subprocess.call(['gsutil','cp',self.parent_weights,tmpdirname])
                the_weights = os.path.join( tmpdirname, os.path.basename(self.parent_weights) )
            else:
                the_weights = self.parent_weights
            self.transferred_keys = self.attach_parent_weights(the_weights)
        
    def on_train_epoch_start(self):
        print(f'starting epoch {self.current_epoch}')
        for name, p in self.named_parameters():
            if self.current_epoch < self.frozen_epochs:
                if name in self.transferred_keys:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            else:
                p.requires_grad = True            

class CNNTransferLearningActivityBias(CNNTransferLearning):
    
    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--parent_weights', type=str, required=True)
        group.add_argument('--frozen_epochs', type=int, default=9999)
        group.add_argument('--rebalance_quantile', type=float)
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')        
        
        return parser
    
    #######################
    # Dead __init__ block #
    #######################
    
    def __init__(self, parent_weights, unfreeze_epoch=9999, 
                 rebalance_quantile=0.5, 
                 optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        super().__init__(parent_weights, frozen_epochs, 
                         optimizer, scheduler, scheduler_monitor, 
                         scheduler_interval, optimizer_args, scheduler_args)

        self.rebalance_quantile = rebalance_quantile
        
    #############
    # PTL hooks #
    #############
        
    def setup(self, stage='training'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            if 'gs://' in self.parent_weights:
                subprocess.call(['gsutil','cp',self.parent_weights,tmpdirname])
                the_weights = os.path.join( tmpdirname, os.path.basename(self.parent_weights) )
            else:
                the_weights = self.parent_weights
            self.transferred_keys = self.attach_parent_weights(the_weights)
            
        train_labs = torch.cat([ batch[1].cpu() for batch in self.train_dataloader() ], dim=0)
        
        assert (self.rebalance_quantile > 0.) and (self.rebalance_quantile < 1.)
        
        self.rebalance_cutoff = torch.quantile(train_labs.max(dim=1)[0], self.rebalance_quantile).item()
        print(f"Activity at {self.rebalance_quantile} quantile: {self.rebalance_cutoff}",file=sys.stderr)
        self.upper_factor     = self.rebalance_quantile / (1. - self.rebalance_quantile)
        self.lower_factor     = self.upper_factor ** -1
        
        self.upper_factor = self.upper_factor
        self.lower_factor = self.lower_factor
        
    def training_step(self, batch, batch_idx):
        x, y   = batch
        y_hat  = self(x)
        
        get_upper = y.max(dim=1)[0].ge( self.rebalance_cutoff )
        get_lower = ~get_upper
        
        if get_upper.sum():
            loss = self.criterion(y_hat[get_upper], y[get_upper]) \
                     .div(get_upper.numel()).mul(get_upper.sum())\
                     .mul( self.upper_factor )
            if get_lower.sum():
                loss += self.criterion(y_hat[get_lower], y[get_lower]) \
                          .div(get_lower.numel()).mul(get_lower.sum())\
                          .mul( self.lower_factor )
        else:
            loss = self.criterion(y_hat, y) \
                     .mul( self.lower_factor )
        
        self.log('train_loss', loss)
        return loss
