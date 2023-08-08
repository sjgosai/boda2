import sys
import argparse
import tempfile
import time
import gzip
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset, Dataset

from ..common import constants, utils

def alphabet_onehotizer(seq, alphabet):
    
    char_array = np.expand_dims( np.array([*seq]), 0 )
    alph_array = np.expand_dims( np.array(alphabet), 1 )
    
    return char_array == alph_array

class OneHotSlicer(nn.Module):
    
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.register_buffer('weight', self.set_weight(in_channels, kernel_size))
        
    def set_weight(self, in_channels, kernel_size):
        outter_cat = []
        for i in range(in_channels):
            inner_stack = [ torch.zeros((kernel_size,kernel_size)) for x in range(in_channels) ]
            inner_stack[i] = torch.eye(kernel_size)
            outter_cat.append( torch.stack(inner_stack, dim=1) )
        return torch.cat(outter_cat, dim=0)
    
    def forward(self, input):
        hook = F.conv1d(input, self.weight)
        hook = hook.permute(0,2,1).flatten(0,1) \
                 .unflatten(1,(self.in_channels, self.kernel_size))
        return hook

class Fasta:
    
    def __init__(self, fasta_path, all_upper=True, 
                 alphabet=constants.STANDARD_NT):
        self.fasta_path = fasta_path
        self.all_upper = all_upper
        self.alphabet = alphabet
        self.read_fasta()
        
    def read_fasta(self):
        
        self.fasta = {}
        self.contig_lengths   = {}
        self.contig_index2key = {}
        self.contig_key2index = {}
        self.contig_descriptions = {}
        
        print('pre-reading fasta into memory', file=sys.stderr)
        with open(self.fasta_path, 'r') as f:
            fa = np.array(
                [ x.rstrip() for x in tqdm.tqdm(f.readlines()) ]
            )
            print('finding keys', file=sys.stderr)
            fa_idx = np.where( np.char.startswith(fa, '>') )[0]
            print('parsing', file=sys.stderr)
            
            for idx, contig_loc in tqdm.tqdm(list(enumerate(fa_idx))):
                contig_info = fa[contig_loc][1:]
                contig_key, *contig_des = contig_info.split()
                
                start_block = fa_idx[idx] + 1
                try:
                    end_block = fa_idx[idx+1]
                except IndexError:
                    end_block = None
                    
                get_blocks = fa[start_block:end_block]
                if self.all_upper:
                    contig_seq = ''.join( np.char.upper(get_blocks) )
                else:
                    contig_seq = ''.join( get_blocks )

                self.fasta[contig_key] = alphabet_onehotizer(
                    contig_seq, self.alphabet
                )
                self.contig_lengths[contig_key] = len(contig_seq)
                self.contig_index2key[idx] = contig_key
                self.contig_key2index[contig_key] = idx
                self.contig_descriptions = contig_des
                    
        print('done',file=sys.stderr)


class FastaDataset(Dataset):
    
    def __init__(self, 
                 fasta_obj, window_size, step_size, 
                 reverse_complements=True,
                 alphabet=constants.STANDARD_NT,
                 complement_dict=constants.DNA_COMPLEMENTS,
                 pad_final=False):
        
        super().__init__()
        
        assert step_size <= window_size, "Gaps will form if step_size > window_size"
        
        self.fasta = fasta_obj
        self.window_size = window_size
        self.step_size = step_size
        
        self.reverse_complements = reverse_complements
        
        self.alphabet = alphabet
        self.complement_dict = complement_dict
        self.complement_matrix = self.parse_complements()
        
        self.pad_final  = pad_final
        
        self.n_keys = len(self.fasta.keys())
        self.key_lens =  { k: self.fasta[k].shape[-1] for k in self.fasta.keys() }
        self.key_n_windows = self.count_windows()
        self.key_rolling_n = np.cumsum([ self.key_n_windows[k] for k in self.fasta.keys() ])
        
        self.key2idx  = { k:i for i,k in enumerate(self.fasta.keys()) }
        self.idx2key  = list(self.fasta.keys())
        
        self.n_unstranded_windows = sum( self.key_n_windows.values() )
                    
    def count_windows(self):
        
        key_n_windows = {}
        
        for k, v in self.key_lens.items():
            
            if v >= self.window_size:
                n = 1
                n += (v - self.window_size) // self.step_size
                if self.pad_final:
                    n += 1 if (v - self.window_size) % self.step_size > 0 else 0
                
            else:
                n = 0
                
            key_n_windows[k] = n
        
        return key_n_windows
        
    def get_fasta_coords(self, idx):
        
        k_id = self.n_keys - sum(self.key_rolling_n > idx)
        n_past = 0 if k_id == 0 else self.key_rolling_n[k_id-1]
        window_idx = idx - n_past
        
        k = self.idx2key[k_id]
        start = window_idx * self.step_size
        end   = min(start + self.window_size, self.key_lens[k])
        start = end - self.window_size
        
        return {'key': k, 'start': start, 'end': end}

    def parse_complements(self):
        
        comp_mat = np.zeros( (len(self.alphabet),len(self.alphabet)) )
        
        for i in range(len(self.alphabet)):
            target_index = self.alphabet.index( self.complement_dict[ self.alphabet[i] ] )
            comp_mat[target_index,i] = 1
        return comp_mat
    
    def __len__(self):
        
        strands = 2 if self.reverse_complements else 1
        
        return self.n_unstranded_windows * strands
    
    def __getitem__(self, idx):
        
        if self.reverse_complements:
            strand = 1 if idx % 2 == 0 else -1
            u_idx = idx // 2
        else:
            u_idx = idx
            strand = 1
        
        fasta_loc = self.get_fasta_coords(u_idx)
        k, start, end = [fasta_loc[x] for x in ['key', 'start', 'end']]
        
        fasta_seq = self.fasta[k][:,start:end].astype(np.float32)
        fasta_seq = fasta_seq if strand == 1 else np.flip( self.complement_matrix @ fasta_seq, axis=-1)
        fasta_seq = torch.tensor(fasta_seq.copy())
        
        loc_tensor= torch.tensor([self.key2idx[k], start, end, strand])
        
        return loc_tensor, fasta_seq

class VCF:
    
    def __init__(self, 
                 vcf_path, 
                 max_allele_size=10000,
                 max_indel_size=10000,
                 alphabet=constants.STANDARD_NT, 
                 strict=False, 
                 all_upper=True, chr_prefix='', 
                 verbose=False
                ):
        
        self.vcf_path = vcf_path
        self.max_allele_size = max_allele_size
        self.max_indel_size = max_indel_size
        self.alphabet = [ x.upper() for x in alphabet ] if all_upper else alphabet
        self.strict   = strict
        self.all_upper= all_upper
        self.chr_prefix = chr_prefix
        self.verbose = verbose
        
        self.vcf = self._open_vcf()
        #self.read_vcf()
        
    def _open_vcf(self):
        
        vcf_colnames = ['chrom','pos','id','ref','alt','qual','filter','info']
        re_pat = matcher = f'[^{"".join(self.alphabet)}]'
        
        # Loading to DataFrame
        print('loading DataFrame', file=sys.stderr)
        if self.vcf_path.endswith('gz'):
            data = pd.read_csv(self.vcf_path, sep='\t', comment='#', header=None, compression='gzip', usecols=[0,1,2,3,4])
        else:
            data = pd.read_csv(self.vcf_path, sep='\t', comment='#', header=None, usecols=[0,1,2,3,4])
        
        data.columns = vcf_colnames[:data.shape[1]]
        data['chrom']= self.chr_prefix + data['chrom'].astype(str)
        
        # Checking and filtering tokens
        print('Checking and filtering tokens', file=sys.stderr)
        if self.all_upper:
            data['ref'] = data['ref'].str.upper()
            data['alt'] = data['alt'].str.upper()
        
        ref_filter = data['ref'].str.contains(re_pat,regex=True)
        alt_filter = data['alt'].str.contains(re_pat,regex=True)
        
        if self.strict:
            assert ref_filter.sum() > 0, "Found unknown token in ref. Abort."
            assert alt_filter.sum() > 0, "Found unknown token in alt. Abort."
        else:
            total_filter = ~(ref_filter | alt_filter)
            data = data.loc[ total_filter ]
        
        # Length checks
        print('Allele length checks', file=sys.stderr)
        ref_lens = data['ref'].str.len()
        alt_lens = data['alt'].str.len()
        
        max_sizes   = np.maximum(ref_lens, alt_lens)
        indel_sizes = np.abs(ref_lens - alt_lens)
        
        size_filter = (max_sizes < self.max_allele_size) & (indel_sizes < self.max_indel_size)
        data = data.loc[size_filter]
        
        print('Done', file=sys.stderr)
        return data.reset_index(drop=True)
        
    def __call__(self, loc_idx=None, iloc_idx=None):
        
        assert (loc_idx is None) ^ (iloc_idx is None), "Use loc XOR iloc"
        
        if loc_idx is not None:
            record = self.vcf.loc[loc_idx]
        else:
            record = self.vcf.iloc[iloc_idx]
            
        return record
        
    
class VcfDataset(Dataset):
    
    def __init__(self, 
                 vcf_obj, fasta_obj, window_size, 
                 relative_start, relative_end,  
                 step_size=1,
                 reverse_complements=True,
                 left_flank='', right_flank='', 
                 all_upper=True, use_contigs=[],
                 alphabet=constants.STANDARD_NT,
                 complement_dict=constants.DNA_COMPLEMENTS):
        
        super().__init__()
        
        self.vcf   = vcf_obj
        self.fasta = fasta_obj
        self.window_size = window_size
        self.relative_start = relative_start
        self.relative_end   = relative_end
        self.grab_size = self.window_size-self.relative_start+self.relative_end-1
        
        self.step_size = step_size
        self.reverse_complements = reverse_complements
        
        self.left_flank = left_flank
        self.right_flank= right_flank
        self.all_upper = all_upper
        self.use_contigs = use_contigs
        self.alphabet = alphabet
        self.complement_dict = complement_dict
        self.complement_matrix = torch.tensor( self.parse_complements() ).float()
        
        self.window_slicer = OneHotSlicer(len(alphabet), window_size)
        
        self.filter_vcf()

    def parse_complements(self):
        
        comp_mat = np.zeros( (len(self.alphabet),len(self.alphabet)) )
        
        for i in range(len(self.alphabet)):
            target_index = self.alphabet.index( self.complement_dict[ self.alphabet[i] ] )
            comp_mat[target_index,i] = 1
        return comp_mat
    
    def encode(self, allele):
        
        my_allele = allele.upper() if self.all_upper else allele
        return alphabet_onehotizer(my_allele, self.alphabet)
        
    '''
    def read_vcf(self):
        
        f = self._open_vcf()
        for i, line in tqdm.tqdm(f.iterrows(),total=f.shape[0]):
            chrom, pos, tag, ref, alt, *others = line

            ref = self.encode(ref)
            alt = self.encode(alt)

            if np.abs(ref.shape[1]-alt.shape[1]) > self.max_indel_size:
                if self.verbose: print(f"skipping large indel at line {i}, id: {tag}", file=sys.stderr)
                continue

            if ref.shape[1] > self.max_allele_size or alt.shape[1] > self.max_allele_size:
                if self.verbose: print(f"skipping large allele at line {i}, id: {tag}", file=sys.stderr)
                continue

            cat_alleles = np.concatenate([ref,alt],axis=1)

            if cat_alleles.sum() == cat_alleles.shape[1]:                
                self.vcf.append(
                    {
                        'chrom': self.chr_prefix + str(chrom),
                        'pos': int(pos),
                        'tag': tag,
                        'ref': ref,
                        'alt': alt,
                        'additional': others,
                    }
                )
            elif self.strict:
                raise ValueError(f"malformed record at line {i}, id: {tag}\nExiting.")
            else:
                print(f"skipping malformed record at line {i}, id: {tag}", file=sys.stderr)
                print(line, file=sys.stderr)
                    
        return None
    '''
    
    def filter_vcf(self):
        pre_len = self.vcf.shape[0]
        
        contig_filter = self.vcf['chrom'].isin(self.fasta.keys())
        print(f"{contig_filter.sum()}/{pre_len} records have matching contig in FASTA", file=sys.stderr)
        if len(self.use_contigs) > 0:
            contig_filter = contig_filter & self.vcf['chrom'].isin(self.use_contigs)
            print(f"removing {np.sum(~self.vcf['chrom'].isin(self.use_contigs))}/{pre_len} records based on contig blacklist", file=sys.stderr)
            
        if contig_filter.sum() < 1:
            print('No contigs passed. Check filters.', file=sys.stderr)
        
        self.vcf = self.vcf.loc[ contig_filter ]
        print(f"returned {self.vcf.shape[0]}/{pre_len} records", file=sys.stderr)
        return None
    
    def __len__(self):
        return self.vcf.shape[0]
    
    def __getitem__(self, idx):
        record = self.vcf.iloc[idx]
        
        ref = self.encode(record['ref'])
        alt = self.encode(record['alt'])
        
        var_loc = record['pos'] - 1
        start   = var_loc - self.relative_end + 1

        trail_start = var_loc + ref.shape[1]
        trail_end   = start + self.grab_size

        len_dif = alt.shape[1] - ref.shape[1]
        start_adjust = len_dif // 2
        end_adjust   = len_dif - start_adjust

        try:
            # Collect reference
            contig = self.fasta[ record['chrom'] ]
            leader = contig[:, start:var_loc]
            trailer= contig[:, trail_start:trail_end]
            
            ref_segments = [leader, ref, trailer]
            
            # Collect alternate
            leader = contig[:, start+start_adjust:var_loc]
            trailer= contig[:, trail_start:trail_end-end_adjust]
            
            alt_segments = [leader, alt, trailer]
            
            # Combine segments
            ref = np.concatenate(ref_segments, axis=-1)
            alt = np.concatenate(alt_segments, axis=-1)
            
            ref = torch.tensor(ref[np.newaxis].astype(np.float32))
            alt = torch.tensor(alt[np.newaxis].astype(np.float32))

            ref_slices = self.window_slicer(ref)[::self.step_size]
            alt_slices = self.window_slicer(alt)[::self.step_size]


            if self.reverse_complements:
                ref_rc = torch.flip(self.complement_matrix @ ref_slices, dims=[-1])
                ref_slices = torch.cat([ref_slices,ref_rc], dim=0)

                alt_rc = torch.flip(self.complement_matrix @ alt_slices, dims=[-1])
                alt_slices = torch.cat([alt_slices,alt_rc], dim=0)

            return {'ref': ref_slices, 'alt': alt_slices}

        except KeyError:
            print(f"No contig: {record['chrom']} in FASTA, skipping", file=sys.stderr)
            return {'ref': None, 'alt': None}