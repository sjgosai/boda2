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
from boda.common import constants
from boda.common.utils import unpack_artifact, model_fn

class FlankBuilder(nn.Module):
    """
    A module for adding flanks to input samples.

    Args:
        left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
        right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
        batch_dim (int, optional): Batch dimension. Default is 0.
        cat_axis (int, optional): Axis along which tensors will be concatenated. Default is -1.

    Attributes:
        left_flank (torch.Tensor): Left flank tensor.
        right_flank (torch.Tensor): Right flank tensor.
        batch_dim (int): Batch dimension.
        cat_axis (int): Axis along which tensors will be concatenated.

    Methods:
        add_flanks(my_sample):
            Adds flanks to a given input sample.

        forward(my_sample):
            Forward pass of the FlankBuilder module.
    """
    
    def __init__(self,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1
                ):
        """
        Initialize a FlankBuilder module.

        Args:
            left_flank (torch.Tensor, optional): Left flank tensor. Default is None.
            right_flank (torch.Tensor, optional): Right flank tensor. Default is None.
            batch_dim (int, optional): Batch dimension. Default is 0.
            cat_axis (int, optional): Axis along which tensors will be concatenated. Default is -1.

        Returns:
            None
        """
        super().__init__()
        
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.batch_dim = batch_dim
        self.cat_axis  = cat_axis
        
    def add_flanks(self, my_sample):
        """
        Adds flanks to a given input sample.

        Args:
            my_sample (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added flanks.
        """
        *batch_dims, channels, length = my_sample.shape
        
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.expand(*batch_dims, -1, -1) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.expand(*batch_dims, -1, -1) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self, my_sample):
        """
        Forward pass of the FlankBuilder module.

        Args:
            my_sample (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.add_flanks(my_sample)

class Mutagenizer(nn.Module):
    """
    A module for generating mutated sequences.

    Args:
        seq_len (int): Length of the input sequence.
        alphabet (str or list of str, optional): Alphabet for the input sequence. Default is constants.STANDARD_NT.

    Attributes:
        n_tokens (int): Number of tokens in the alphabet.
        clear_midpoint (torch.Tensor): Tensor for clearing the midpoint.
        set_variants (torch.Tensor): Tensor for setting variants.
        windower (boda.data.fasta_datamodule.OneHotSlicer): OneHotSlicer instance for sequence windowing.

    Methods:
        forward(input):
            Forward pass of the Mutagenizer module.
    """
    def __init__(self, seq_len, alphabet=constants.STANDARD_NT):
        """
        Initialize a Mutagenizer module.

        Args:
            seq_len (int): Length of the input sequence.
            alphabet (str or list of str, optional): Alphabet for the input sequence. Default is constants.STANDARD_NT.

        Returns:
            None
        """
        super().__init__()
        
        self.n_tokens = len(alphabet)
        
        clear_midpoint = torch.ones((self.n_tokens, self.n_tokens, seq_len*2-1))
        clear_midpoint[:,:,seq_len-1] = 0
        set_variants   = torch.zeros_like(clear_midpoint)
        set_variants[:,:,seq_len-1] = torch.eye(self.n_tokens)
        
        self.register_buffer('clear_midpoint', clear_midpoint)
        self.register_buffer('set_variants', set_variants)
        
        self.windower = boda.data.fasta_datamodule.OneHotSlicer(self.n_tokens,seq_len)
        
    def forward(self, input):
        """
        Forward pass of the Mutagenizer module.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Mutated sequence tensor.
        """
        hook = input.unsqueeze(-3)
        shape = list(hook.shape)
        shape[-3] = shape[-2]
        hook = hook.expand(*shape)
        hook = hook * self.clear_midpoint
        hook = hook + self.set_variants
        
        return self.windower( hook.flatten(0,-3) )


def main(args):
    """
    Execute the main functionality of the script.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.

    Returns:
        None
    """
    
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
    
    ###################
    ## Setup helpers ##
    ###################
    left_flank = boda.common.utils.dna2tensor( 
        args.left_flank 
    ).unsqueeze(0)
    left_flank.shape

    right_flank= boda.common.utils.dna2tensor( 
        args.right_flank 
    ).unsqueeze(0)
    right_flank.shape

    flank_builder = FlankBuilder(
        left_flank=left_flank,
        right_flank=right_flank,
    )
    
    mutagenizer = Mutagenizer(args.sequence_length)
    
    #################
    ## Setup FASTA ##
    #################
    fasta_dict = boda.data.Fasta(args.fasta_file)
    n_tokens = len(fasta_dict.alphabet)
    
    fasta_data = boda.data.FastaDataset(
        fasta_dict.fasta, 
        window_size=args.sequence_length*2-1, step_size=1, 
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
    
    current_contig = ''

    f = h5py.File(args.output,'w')
    h5_datasets = [ f.create_dataset(key, (fasta_data.key_lens[key], 4, 3), dtype=np.float16) for key in fasta_data.idx2key ]
    for dset in h5_datasets:
        dset.set_fill_value = np.nan
        
    first_chr, first_start, first_end, *first_extra  = list(fasta_subset[0][0])
    last_chr, last_start, last_end, *last_extra      = list(fasta_subset[-1][0])
    
    process_span = [fasta_data.idx2key[first_chr], first_start, first_end, fasta_data.idx2key[last_chr], last_start, last_end]
    print("Processing intervals {} {} {} to {} {} {}".format(*process_span), file=sys.stderr)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, batch in enumerate(tqdm.tqdm(fasta_loader)):
                
                location, sequence = [ y.contiguous() for y in batch ]
                
                current_bsz = location.shape[0]
                
                mutated = mutagenizer(sequence)
                forward = flank_builder(mutated)
                revcomp = flank_builder(mutated.flip(dims=(1,2)))
                
                result = my_model(forward.cuda()).div(2.) + my_model(revcomp.cuda()).div(2.)
                
                result = result.unflatten(0,(current_bsz, n_tokens, args.sequence_length)).mean(dim=2)
                
                for (chrom_idx, start, end, strand), position_activity in zip(location, result):
                    f[ fasta_data.idx2key[chrom_idx] ][start+args.sequence_length-1] = position_activity.cpu().half().numpy()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Saturation mutagenesis tool.")
    parser.add_argument('--artifact_path', type=str, required=True, help='Pre-trained model artifacts.')
    parser.add_argument('--fasta_file', type=str, required=True, help='FASTA reference file.')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 path.')
    parser.add_argument('--job_id', type=int, default=0, help='Job partition index for distributed computing.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Total number of job partitions.')
    parser.add_argument('--sequence_length', type=int, default=200, help='Length of DNA sequence to test during mutagenesis.')
    parser.add_argument('--left_flank', type=str, default=boda.common.constants.MPRA_UPSTREAM[-200:], help='Upstream padding.')
    parser.add_argument('--right_flank', type=str, default=boda.common.constants.MPRA_DOWNSTREAM[:200], help='Downstream padding.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during sequence extraction from FASTA.')
    args = parser.parse_args()
    
    main(args)
