import os
import sys
import re
import copy
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

def save_proposals(proposals, args):
    """
    Save the proposals and associated arguments to a file.

    This function saves the generated proposals and the input arguments used for their generation
    to a file. The file is named based on the current timestamp and a random tag. The saved file can
    be placed in a local directory or on Google Cloud Storage if a 'gs://' path is provided in the arguments.

    Args:
        proposals (list): List of proposals.
        args (dict): Dictionary of input arguments.

    Returns:
        None
    """
    save_dict = {
        'proposals': proposals,
        'args'     : args,
        'timestamp'    : time.strftime("%Y%m%d_%H%M%S"),
        'random_tag'   : random.randint(100000,999999)
    }
    filename = f'proposals__{save_dict["timestamp"]}__{save_dict["random_tag"]}.pt'
    torch.save(save_dict, filename)
        
    if 'gs://' in args['Main args'].proposal_path:
        clound_target = os.path.join(args['Main args'].proposal_path,filename)
        subprocess.check_call(
            ['gsutil', 'cp', filename, clound_target]
        )
    else:
        os.makedirs(args['Main args'].proposal_path, exist_ok=True)
        shutil.copy(filename, args['Main args'].proposal_path)
    
    final_loc = os.path.join(args['Main args'].proposal_path,filename)
    print(f'Proposals deposited at:\n\t{final_loc}', file=sys.stderr)

def save_proposals(proposals, args):
    """
    Save the proposals and associated arguments to a file.

    This function saves the generated proposals and the input arguments used for their generation
    to a file. The file is named based on the current timestamp and a random tag. The saved file can
    be placed in a local directory or on Google Cloud Storage if a 'gs://' path is provided in the arguments.

    Args:
        proposals (list): List of proposals.
        args (dict): Dictionary of input arguments.

    Returns:
        None
    """
    save_dict = {
        'proposals': proposals,
        'args'     : args,
        'timestamp'    : time.strftime("%Y%m%d_%H%M%S"),
        'random_tag'   : random.randint(100000,999999)
    }
    filename = f'{args["Main args"].proposal_path}__{save_dict["timestamp"]}__{save_dict["random_tag"]}.pt'
    
    if 'gs://' in args['Main args'].proposal_path:
        torch.save(save_dict, os.path.basename(filename))
        subprocess.check_call(
            ['gsutil', 'cp', os.path.basename(filename), filename]
        )
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(save_dict, filename)
    
    print(f'Proposals deposited at:\n\t{filename}', file=sys.stderr)


def main(args):
    """
    Main function for generating and saving proposals.

    This function orchestrates the generation of proposals using the specified modules and arguments.
    It creates instances of the required modules, generates proposals, and saves them using the `save_proposals`
    function.

    Args:
        args (dict): Dictionary of input arguments.

    Returns:
        tuple: A tuple containing instances of `params_module`, `energy_module`, `generator_module`, and
               a list of generated proposals.
    """
    args_copy = copy.deepcopy(args)
    
    params_module     = getattr(boda.generator.parameters, args['Main args'].params_module)
    energy_module    = getattr(boda.generator.energy    , args['Main args'].energy_module)
    generator_module = getattr(boda.generator           , args['Main args'].generator_module)
    
    params_args     = vars(params_module.process_args(args))
    energy_args    = vars(energy_module.process_args(args))
    generator_args = generator_module.process_args(args)
    generator_constructor_args, generator_runtime_args = [ vars(arg_subset) for arg_subset in generator_args ]
    
    params    = params_module(**params_args)
    energy    = energy_module(**energy_args)
    
    if args['Main args'].penalty_module is not None:
        penalty_module = getattr(boda.generator.energy, args['Main args'].penalty_module)
        energy.__class__ = type(
            'energy_module',
            (energy_module, penalty_module),
            vars(penalty_module.process_args(args))
        )
    else:
        penalty_module = None
    current_penalty = None

    generator_constructor_args['params']    = params
    generator_constructor_args['energy_fn'] = energy
    generator_runtime_args['energy_threshold'] = args['Main args'].energy_threshold
    generator_runtime_args['max_attempts'] = args['Main args'].max_attempts
    generator = generator_module(**generator_constructor_args)
    
    params.cuda()
    energy.cuda()
    
    proposal_sets = []
    for round_id, get_n in enumerate(args['Main args'].n_proposals):
        print(f'Starting round: {round_id}, generate {get_n} proposals', file=sys.stderr)
        generator_runtime_args['n_proposals'] = get_n
        proposal = generator.generate(**generator_runtime_args)
        proposal['penalty'] = current_penalty
        proposal_sets.append(proposal)
        
        #sns.histplot(proposal['energies'].numpy())
        #plt.show()
        
        if args['Main args'].penalty_module is not None:
            current_penalty = energy.update_penalty(proposal)
            
        if args['Main args'].reset_params:
            generator.params = params_module(**params_args)
            
        print('finished round', file=sys.stderr)
            
    save_proposals(proposal_sets, args_copy)
    return params, energy, generator, proposal_sets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BODA generator", add_help=False)
    group = parser.add_argument_group('Main args')

    group.add_argument('--params_module', type=str, required=True, help='')
    group.add_argument('--energy_module', type=str, required=True, help='')    
    group.add_argument('--generator_module', type=str, required=True, help='')
    group.add_argument('--penalty_module', type=str, help='')
    group.add_argument('--monitor', type=str, help='')
    group.add_argument('--n_proposals', nargs='+', type=int, help='')
    group.add_argument('--energy_threshold', type=float, default=float("Inf"))
    group.add_argument('--max_attempts', type=int, default=10000)
    group.add_argument('--reset_params', type=utils.str2bool, default=True)
    group.add_argument('--proposal_path', type=str)

    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False, help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')
    
    known_args, leftover_args = parser.parse_known_args()
    
    Params    = getattr(boda.generator, known_args.params_module)
    Energy    = getattr(boda.generator.energy, known_args.energy_module)
    Generator = getattr(boda.generator, known_args.generator_module)
    
    parser = Params.add_params_specific_args(parser)
    parser = Energy.add_energy_specific_args(parser)
    parser = Generator.add_generator_specific_args(parser)
    
    if known_args.penalty_module is not None:
        Penalty = getattr(boda.generator.energy, known_args.penalty_module)
        parser = Penalty.add_penalty_specific_args(parser)
        
    parser.add_argument('--help', '-h', action='help')
    
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
        
    args = boda.common.utils.organize_args(parser, args)
    
    _ = main(args)