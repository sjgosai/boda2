import sys
import argparse
import tempfile
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd

import pyfaidx

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset, Dataset

from ..common import constants, utils

class AlphabetOnehotizer:
    
    def __init__(self,
                 alphabet=constants.STANDARD_NT,
                 dead_characters=True):
        
        self.alphabet = alphabet
        self.dead_characters = dead_characters
        self.n_tokens = len(alphabet)
        
        if dead_characters:
            n_tokens = len(alphabet)
            def return_constant():
                return n_tokens
            self.tokenizer = defaultdict(return_constant)
            self.tokenizer.update({k:i for i,k in enumerate(alphabet)})
            
        else:
            self.tokenizer = {k:i for i,k in enumerate(alphabet)}
            
    def __call__(self, input_str):
        
        tokens = torch.tensor([self.tokenizer[s] for s in input_str])
        onehot = F.one_hot(tokens, num_classes=len(self.tokenizer.keys())) \
                   .permute(1,0)
        
        return onehot[:self.n_tokens].float()

class FastaDataset(Dataset):
    
    def __init__(self, 
                 fasta_path, window_size, step_size, 
                 left_flank=None, right_flank=None, 
                 pad_final=True):
        
        super().__init__()
        
        assert step_size <= window_size, "Gaps will form if step_size > window_size"
        
        self.fasta = pyfaidx.Fasta(fasta_path, sequence_always_upper=True)
        self.window_size = window_size
        self.step_size = step_size
        
        self.left_flank = left_flank
        self.right_flank= right_flank
        self.pad_final  = pad_final
        
        self.n_keys = len(self.fasta.keys())
        self.key_lens =  { k: len(self.fasta[k]) for k in self.fasta.keys() }
        self.key_n_windows = self.count_windows()
        self.key_rolling_n = np.cumsum([ self.key_n_windows[k] for k in self.fasta.keys() ])
        
        self.key2idx  = { k:i for i,k in enumerate(self.fasta.keys()) }
        self.idx2key  = list(self.fasta.keys())
        
        self.onehotizer = AlphabetOnehotizer()
            
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
        
    def add_flanks(self, x):
        
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank )
            
        pieces.append( x )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank )
            
        return "".join(pieces)
    
    def get_fasta_coords(self, idx):
        
        k_id = self.n_keys - sum(self.key_rolling_n > idx)
        n_past = 0 if k_id == 0 else self.key_rolling_n[k_id-1]
        window_idx = idx - n_past
        
        k = self.idx2key[k_id]
        start = window_idx * self.step_size
        end   = min(start + self.window_size, self.key_lens[k])
        start = end - self.window_size
        
        return {'key': k, 'start': start, 'end': end}
    
    def __len__(self):
        
        return sum( self.key_n_windows.values() )
    
    def __getitem__(self, idx):
        
        fasta_loc = self.get_fasta_coords(idx)

        k, start, end = [fasta_loc[x] for x in ['key', 'start', 'end']]
        fasta_seq = self.fasta[k][start:end].seq
        fasta_seq = self.add_flanks(fasta_seq)
        
        onehot = self.onehotizer(fasta_seq)
        
        loc_tensor= torch.tensor([self.key2idx[k], start, end])
        
        return loc_tensor, onehot            

