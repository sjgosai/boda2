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

import boda
from boda.common import utils
from boda.common.utils import set_best, save_model, unpack_artifact, model_fn

import hypertune

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
    group = parser.add_argument_group('Main args')
    
    group.add_argument('--param_module', type=str, required=True, help='')
    group.add_argument('--generator_module', type=str, required=True, help='')
    group.add_argument('--energy_module', type=str, required=True, help='')
    
    group.add_argument('--penalty_module', type=str, help='')
    
    group.add_argument('--model_artifacts', type=str, default='/opt/ml/checkpoints/', help='')
    group.add_argument('--monitor', type=str, help='')
    
    group.add_argument('--energy_threshold', type=float, help='')
    group.add_argument('--n_samples', type=list, help='')
    
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False, help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')
    
    known_args, leftover_args = parser.parse_known_args()
    
    Param     = getattr(boda.generator, known_args.param_module)
    Generator = getattr(boda.generator, known_args.generator_module)
    Energy    = getattr(boda.generator.energy, known_args.energy_module)
    
    parser = Param.add_param_specific_args(parser)
    parser = Param.add_generator_specific_args(parser)
    parser = Param.add_energy_specific_args(parser)
    
    if known_args.penalty_module is not None:
        Penalty = getattr(boda.generator.energy, known_args.penalty_module)
        parser = Param.add_penalty_specific_args(parser)
        
    parser.add_argument('--help', '-h', action='help')
    
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
        
    args = boda.common.utils.organize_args(parser, args)
    
    main(args)