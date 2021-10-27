import os
import sys
import re
import time
import yaml
import shutil
import argparse
import tarfile
import tempfile
import random
import subprocess

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import boda
from boda.common import utils
from boda.common.utils import set_best, save_model, unpack_artifact, model_fn

import hypertune

def main(args):
    
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module= getattr(boda.model, args['Main args'].model_module)
    graph_module= getattr(boda.graph, args['Main args'].graph_module)

    data = data_module(**vars(data_module.process_args(args)))
    model= model_module(**vars(model_module.process_args(args)))

    model.__class__ = type(
        'BODA_module',
        (model_module,graph_module),
        vars(graph_module.process_args(args))
    )
    
    use_callbacks = {
        'learning_rate_monitor': LearningRateMonitor()
    }
    if args['Main args'].checkpoint_monitor is not None:
        use_callbacks['model_checkpoint'] = ModelCheckpoint(
            save_top_k=1,
            monitor=args['Main args'].checkpoint_monitor, 
            mode=args['Main args'].stopping_mode
        )
        use_callbacks['early_stopping'] = EarlyStopping(
            monitor=args['Main args'].checkpoint_monitor, 
            patience=args['Main args'].stopping_patience,
            mode=args['Main args'].stopping_mode
        )
    
    try:
        AIP_logs = os.environ['AIP_TENSORBOARD_LOG_DIR']
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=AIP_logs
        )
        print(f"Saving logs to AIP provided target: {AIP_logs}")
    except KeyError:
        tb_logger = True
        print("Saving logs to PTL default")
        
    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args['pl.Trainer'], 
        callbacks=list(use_callbacks.values()),
        logger=tb_logger
    )
    
    trainer.fit(model, data)
    
    model = set_best(model, use_callbacks)
    
    try:
        mc_dict = vars(use_callbacks['model_checkpoint'])
        keys = ['monitor', 'best_model_score']
        tag, metric = [ mc_dict[key] for key in keys ]
        model.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=tag,
            metric_value=metric.item(),
            global_step=model.global_step + 1)
        print(f'{tag} at {model.global_step}: {metric}', file=sys.stderr)
    except KeyError:
        print('Used default checkpointing.', file=sys.stderr)
    except AttributeError:
        print("No hypertune instance found.", file=sys.stderr)
        pass
    
    save_model(data_module, model_module, graph_module, 
               model, trainer, args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True, help='BODA data module to process dataset.')
    group.add_argument('--model_module',type=str, required=True, help='BODA model module to fit dataset.')
    group.add_argument('--graph_module',type=str, required=True, help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/', help='Path where model artifacts are deposited.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    group.add_argument('--checkpoint_monitor', type=str, help='String to monior PTL logs if saving best.')
    group.add_argument('--stopping_mode', type=str, default='min', help='Goal for monitored metric e.g. (max or min).')
    group.add_argument('--stopping_patience', type=int, default=100, help='Number of epochs of non-improvement tolerated before early stopping.')
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False, help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')
    known_args, leftover_args = parser.parse_known_args()
    
    Data  = getattr(boda.data,  known_args.data_module)
    Model = getattr(boda.model, known_args.model_module)
    Graph = getattr(boda.graph, known_args.graph_module)
    
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    known_args, leftover_args = parser.parse_known_args()
    
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
    
    args = boda.common.utils.organize_args(parser, args)
    
    main(args)