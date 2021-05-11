import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

from ..common import utils
from .utils import add_optimizer_specific_args, add_scheduler_specific_args, reorg_optimizer_args

class CNNBasicTraining(LightningModule):
    
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
        
    def categorical_mse(self, x, y):
        return (x - y).pow(2).mean(dim=0)
        
    def configure_optimizers(self):
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
        res_str = '| Validation | arithmatic mean loss: {:.5f} | harmonic mean loss: {:.5f} |' \
                    .format(arit_mean, harm_mean)
        print('')
        print('-'*len(res_str))
        print(res_str)
        print('-'*len(res_str))
        print('')
        #print(val_step_outputs)
        #for out in val_step_outputs:
        #    print(out)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)       
        
