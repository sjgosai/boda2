import subprocess
import sys
import os
import shutil
import gzip
import csv
import argparse
import multiprocessing

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import boda
from boda.common import constants, utils
from boda.common.utils import unpack_artifact, model_fn

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("h5py")
import h5py

###################################
## Contribution Scoreing helpers ##
###################################

from torch.utils.data import (random_split, DataLoader, TensorDataset, ConcatDataset)
from torch.distributions.categorical import Categorical

class mpra_predictor(nn.Module):
    def __init__(self,
                 model,
                 pred_idx=0,
                 ini_in_len=200,
                 model_in_len=600,
                 cat_axis=-1,
                 dual_pred=False):
        super().__init__()
        self.model = model
        self.pred_idx = pred_idx
        self.ini_in_len = ini_in_len 
        self.model_in_len = model_in_len
        self.cat_axis = cat_axis  
        self.dual_pred = dual_pred
        
        try: self.model.eval()
        except: pass
        
        self.register_flanks()
    
    def forward(self, x):
        pieces = [self.left_flank.repeat(x.shape[0], 1, 1), x, self.right_flank.repeat(x.shape[0], 1, 1)]
        in_tensor = torch.cat( pieces, axis=self.cat_axis)
        if self.dual_pred:
            dual_tensor = utils.reverse_complement_onehot(in_tensor)
            out_tensor = self.model(in_tensor)[:, self.pred_idx] + self.model(dual_tensor)[:, self.pred_idx]
            out_tensor = out_tensor / 2.0
        else:
            out_tensor = self.model(in_tensor)[:, self.pred_idx]
        return out_tensor
    
    def register_flanks(self):
        missing_len = self.model_in_len - self.ini_in_len
        left_idx = - missing_len//2 + missing_len%2
        right_idx = missing_len//2 + missing_len%2
        left_flank = utils.dna2tensor(constants.MPRA_UPSTREAM[left_idx:]).unsqueeze(0)
        right_flank = utils.dna2tensor(constants.MPRA_DOWNSTREAM[:right_idx]).unsqueeze(0)         
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank) 
        

def isg_contributions(sequences,
                      predictor,
                      num_steps=50,
                      num_samples=20,
                      eval_batch_size=1024,
                      theta_factor=15):
    
    batch_size = eval_batch_size // num_samples
    temp_dataset = TensorDataset(sequences)
    temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_salient_maps = []
    all_gradients = []
    for local_batch in temp_dataloader:
        target_thetas = (theta_factor * local_batch[0].cuda()).requires_grad_()
        line_gradients = []
        for i in range(0, num_steps + 1):
            point_thetas = (i / num_steps * target_thetas)
            point_distributions = F.softmax(point_thetas, dim=-2)

            nucleotide_probs = Categorical(torch.transpose(point_distributions, -2, -1))
            sampled_idxs = nucleotide_probs.sample((num_samples, ))
            sampled_nucleotides_T = F.one_hot(sampled_idxs, num_classes=4)
            sampled_nucleotides = torch.transpose(sampled_nucleotides_T, -2, -1)
            distribution_repeater = point_distributions.repeat(num_samples, *[1 for i in range(3)])
            sampled_nucleotides = sampled_nucleotides - distribution_repeater.detach() + distribution_repeater 
            samples = sampled_nucleotides.flatten(0,1)

            preds = predictor(samples)
            point_predictions = preds.unflatten(0, (num_samples, target_thetas.shape[0])).mean(dim=0)
            point_gradients = torch.autograd.grad(point_predictions.sum(), inputs=point_thetas, retain_graph=True)[0]
            line_gradients.append(point_gradients)
            
        gradients = torch.stack(line_gradients).mean(dim=0).detach()
        all_salient_maps.append(gradients * target_thetas.detach())
        all_gradients.append(gradients)
    return theta_factor * torch.cat(all_gradients).cpu()
    # return torch.cat(all_salient_maps).cpu(), theta_factor * torch.cat(all_gradients).cpu()


def batch_to_contributions(onehot_sequences,
                           model,
                           model_output_len=3,
                           seq_len = 200,
                           eval_batch_size=1040):
    
    extended_contributions = []
    for i in range(model_output_len):
        predictor = mpra_predictor(model=model, pred_idx=i, ini_in_len=seq_len).cuda()
        extended_contributions.append(isg_contributions(onehot_sequences, predictor, eval_batch_size=eval_batch_size))
        
    return torch.stack(extended_contributions, dim=-1)

########################
## Additional helpers ##
########################

def prepare_hdf5_file(fa_dataset, h5_file, subset=None):
    strands = 2 if fa_dataset.reverse_complements else 1
    size = subset if subset is not None else len(fa_dataset)
    
    h5_file.create_dataset('contribution_scores', (size, 4, 200, 3), dtype=np.float16)
    h5_file.create_dataset('locations', (size, 4), dtype=np.int64)
    
    h5_file['contribution_scores'].attrs['axis_names'] = ['windows', 'tokens', 'length', 'cells']
    h5_file['locations'].attrs['column_names'] = ['contig', 'start', 'end', 'strand']
    h5_file['locations'].attrs['contig_keys'] = fa_dataset.idx2key
    
    h5_file['contribution_scores'].set_fill_value = np.nan
        
    return h5_file

    
def main(args):
    
    ##################
    ## Import Model ##
    ##################
    if os.path.isdir('./artifacts'):
        shutil.rmtree('./artifacts')

    unpack_artifact(args.artifact_path)

    model_dir = './artifacts'

    my_model = model_fn(model_dir)
    my_model.cuda()
    my_model.eval()
    
    #################
    ## Setup FASTA ##
    #################
    fasta_dict = boda.data.Fasta(args.fasta_file)
    n_tokens = len(fasta_dict.alphabet)
    
    fasta_data = boda.data.FastaDataset(
        fasta_dict.fasta, 
        window_size=args.sequence_length, step_size=args.step_size, 
        reverse_complements=False
    )
    
    if args.n_jobs > 1:
        extra_tasks = len(fasta_data) % args.n_jobs
        if extra_tasks > 0:
            subset_size = (len(fasta_data)-extra_tasks) // (args.n_jobs-1)
        else:
            subset_size = len(fasta_data) // args.n_jobs
        start_idx = subset_size*args.job_id
        stop_idx  = min(len(fasta_data), subset_size*(args.job_id+1))
        fasta_subset = torch.utils.data.Subset(fasta_data, np.arange(start_idx, stop_idx))
    else:
        fasta_subset = fasta_data
    
    fasta_loader = torch.utils.data.DataLoader(fasta_subset, batch_size=args.batch_size, shuffle=False)
    
    f = h5py.File(args.output,'w')
    f = prepare_hdf5_file(fasta_data, f, subset=stop_idx-start_idx)
    
    first_chr, first_start, first_end, *first_extra  = list(fasta_subset[0][0])
    last_chr, last_start, last_end, *last_extra      = list(fasta_subset[-1][0])
    
    process_span = [fasta_data.idx2key[first_chr], first_start, first_end, fasta_data.idx2key[last_chr], last_start, last_end]
    print("Processing intervals {} {} {} to {} {} {}".format(*process_span), file=sys.stderr)
    
    h5_start = 0
    for i, batch in enumerate(tqdm.tqdm(fasta_loader)):

        location, sequence = [ y.contiguous() for y in batch ]

        current_bsz = location.shape[0]
        
        results = batch_to_contributions(sequence, my_model, 
                                         model_output_len=3, 
                                         seq_len = args.sequence_length, 
                                         eval_batch_size=args.internal_batch_size)
        
        f['locations']          [h5_start:h5_start+current_bsz] = location
        f['contribution_scores'][h5_start:h5_start+current_bsz] = results

        h5_start = h5_start+current_bsz

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Contribution scoring tool.")
    parser.add_argument('--artifact_path', type=str, required=True, help='Pre-trained model artifacts.')
    parser.add_argument('--fasta_file', type=str, required=True, help='FASTA reference file.')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 path.')
    parser.add_argument('--job_id', type=int, default=0, help='Job partition index for distributed computing.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Total number of job partitions.')
    parser.add_argument('--sequence_length', type=int, default=200, help='Length of DNA sequence to test during mutagenesis.')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for windows to be used for contribution scoring.')
    parser.add_argument('--left_flank', type=str, default=boda.common.constants.MPRA_UPSTREAM[-200:], help='Upstream padding.')
    parser.add_argument('--right_flank', type=str, default=boda.common.constants.MPRA_DOWNSTREAM[:200], help='Downstream padding.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during sequence extraction from FASTA.')
    parser.add_argument('--internal_batch_size', type=int, default=1040, help='Internal batch size for contribution scoring.')
    args = parser.parse_args()
    
    main(args)