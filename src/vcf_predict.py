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

class VepTester(nn.Module):
    
    def __init__(self,
                  model
                 ):
        
        super().__init__()
        self.use_cuda = torch.cuda.device_count() >= 1
        self.model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        
    def forward(self, ref_batch, alt_batch):
        
        ref_shape, alt_shape = ref_batch.shape, alt_batch.shape
        #(batch_size, n_windows, n_tokens, length)
        #n_windows = (2 if reverse_complments else 1) * (relative_end - relative_start) // step_size
        assert ref_shape == alt_shape
        
        ref_batch = ref_batch.flatten(0,1)
        alt_batch = alt_batch.flatten(0,1)
        
        with torch.cuda.amp.autocast():
            ref_preds = self.model(ref_batch.contiguous())
            alt_preds = self.model(alt_batch.contiguous())

        ref_preds = ref_preds.unflatten(0, ref_shape[0:2])
        ref_preds = ref_preds.unflatten(1, (2, ref_shape[1]//2))
        
        alt_preds = alt_preds.unflatten(0, alt_shape[0:2])
        alt_preds = alt_preds.unflatten(1, (2, alt_shape[1]//2))
            
        skew_preds = alt_preds - ref_preds

        return {'ref': ref_preds, 
                'alt': alt_preds, 
                'skew': skew_preds}
    

def main(args):
    USE_CUDA = torch.cuda.device_count() >= 1
    print(sys.argv)
    print(f"contig list len test: {len(args.use_contigs)}")
    ##################
    ## Import Model ##
    ##################
    if os.path.isdir('./artifacts'):
        shutil.rmtree('./artifacts')

    unpack_artifact(args.artifact_path)

    model_dir = './artifacts'

    my_model = model_fn(model_dir)
    my_model.eval()
    if USE_CUDA:
        my_model.cuda()
    
    #################
    ## Setup FASTA ##
    #################
    fasta_data = boda.data.Fasta(args.fasta_file)
    
    vcf = boda.data.VCF(
        args.vcf_file, chr_prefix=args.vcf_contig_prefix, 
        max_allele_size=20, max_indel_size=20,
    )
    
    WINDOW_SIZE = 200
    RELATIVE_START = args.relative_start
    RELATIVE_END = args.relative_end
    
    vcf_data = boda.data.VcfDataset(
        vcf.vcf, fasta_data.fasta, WINDOW_SIZE, 
        RELATIVE_START, RELATIVE_END, step_size=args.step_size, 
        left_flank='', right_flank='', use_contigs=args.use_contigs,
    )

    if args.n_jobs > 1:
        extra_tasks = len(vcf_data) % args.n_jobs
        if extra_tasks > 0:
            subset_size = (len(vcf_data)-extra_tasks) // (args.n_jobs-1)
        else:
            subset_size = len(vcf_data) // args.n_jobs
        start_idx = subset_size*args.job_id
        stop_idx  = min(len(vcf_data), subset_size*(args.job_id+1))
        vcf_subset = torch.utils.data.Subset(vcf_data, np.arange(start_idx, stop_idx))
        vcf_table  = vcf_data.vcf.iloc[start_idx:stop_idx]
    else:
        vcf_subset = vcf_data
        vcf_table  = vcf_data.vcf

    
    vcf_loader = torch.utils.data.DataLoader( vcf_subset, batch_size=args.batch_size*max(1,torch.cuda.device_count()) )
    
    left_flank = boda.common.utils.dna2tensor( 
        args.left_flank 
    ).unsqueeze(0).unsqueeze(0)

    right_flank= boda.common.utils.dna2tensor( 
        args.right_flank
    ).unsqueeze(0).unsqueeze(0)
    
    flank_builder = utils.FlankBuilder(
        left_flank=left_flank,
        right_flank=right_flank,
    )
    if USE_CUDA:
        flank_builder.cuda()
    
    vep_tester = VepTester(my_model)
    
    ref_preds = []
    alt_preds = []
    skew_preds= []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(vcf_loader)):
            ref_allele, alt_allele = batch['ref'], batch['alt']
            
            if USE_CUDA:
                ref_allele = ref_allele.cuda()
                alt_allele = alt_allele.cuda()

            ref_allele = flank_builder(ref_allele).contiguous()
            alt_allele = flank_builder(alt_allele).contiguous()

            all_preds = vep_tester(ref_allele, alt_allele)

            ref_preds.append(all_preds['ref'].cpu())
            alt_preds.append(all_preds['alt'].cpu())
            skew_preds.append(all_preds['skew'].cpu())

    ref_preds = torch.cat(ref_preds, dim=0)
    alt_preds = torch.cat(alt_preds, dim=0)
    skew_preds= torch.cat(skew_preds, dim=0)

    if not args.raw_predictions:
        skew_preds =  getattr(torch, args.reduction)(skew_preds.flatten(1,2), dim=1)        
        pd.concat([ vcf_table, pd.DataFrame(skew_preds.numpy()) ], axis=1) \
          .to_csv(args.output, sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    else:
        torch.save({'ref': ref_preds, 'alt': alt_preds, 'vcf': vcf_table}, args.output)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Contribution scoring tool.")
    # Input info
    parser.add_argument('--artifact_path', type=str, required=True, help='Pre-trained model artifacts.')
    parser.add_argument('--vcf_file', type=str, required=True, help='Variants to test in VCF format.')
    parser.add_argument('--fasta_file', type=str, required=True, help='FASTA reference file.')
    # Output info
    parser.add_argument('--output', type=str, required=True, help='Output path. Simple VCF.')
    parser.add_argument('--raw_predictions', type=utils.str2bool, default=False, help='Dump raw ref/alt predictions as tensors.')
    # Data preprocessing
    parser.add_argument('--left_flank', type=str, default=boda.common.constants.MPRA_UPSTREAM[-200:], help='Upstream padding.')
    parser.add_argument('--right_flank', type=str, default=boda.common.constants.MPRA_DOWNSTREAM[:200], help='Downstream padding.')
    parser.add_argument('--vcf_contig_prefix', type=str, default='', help='Prefix to append VCF contig IDs to match FASTA contig IDs.')
    # VEP testing conditions
    parser.add_argument('--relative_start', type=int, default=0, help='Leftmost position where variant is tested.')
    parser.add_argument('--relative_end', type=int, default=200, help='Rightmost position where variant is tested.')
    parser.add_argument('--step_size', type=int, default=1, help='Step size between positions where variants are tested.')
    parser.add_argument('--reduction', type=str, default='mean', help='Specify reduction over testing conditions. Must be a reducing function in base torch.')
    # Throughput management
    parser.add_argument('--use_contigs', type=list, nargs='*', default=[], help='Optional list of contigs (space seperated) to restrict testing to.')    
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during sequence extraction from FASTA.')
    parser.add_argument('--job_id', type=int, default=0, help='Job partition index for distributed computing.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Total number of job partitions.')
    args = parser.parse_args()
    
    main(args)