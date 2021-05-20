import sys
import argparse
import tempfile
from functools import partial

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset

from ..common import constants, utils
        

class MPRA_DataModule(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        
        group.add_argument('--datafile_path', type=str, required=True)
        group.add_argument('--data_project', type=str, nargs='+', default=['BODA', 'UKBB'])
        group.add_argument('--project_column', type=str, default='data_project')
        group.add_argument('--sequence_column', type=str, default='nt_sequence')
        group.add_argument('--activity_columns', type=str, nargs='+', default=['K562', 'HepG2.neon', 'SKNSH'])
        group.add_argument('--valtest_chrs', type=str, nargs='+', default={'7','13','17','19','21','X'})
        group.add_argument('--chr_column', type=str, default='chr')
        group.add_argument('--batch_size', type=int, default=32, 
                           help='Number of examples in each mini batch')         
        group.add_argument('--padded_seq_len', type=int, default=600, 
                           help='Desired total sequence length after padding') 
        group.add_argument('--num_workers', type=int, default=8, 
                           help='number of gpus or cpu cores to be used') 
        group.add_argument('--normalize', type=utils.str2bool, default=False, 
                           help='apply standard score normalization') 
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        data_args    = grouped_args['Data Module args']
        return data_args

    def __init__(self,
                 datafile_path,
                 data_project=['BODA', 'UKBB'],
                 project_column='data_project',
                 sequence_column='nt_sequence',
                 activity_columns=['K562_mean', 'HepG2_mean', 'SKNSH_mean'],
                 valtest_chrs={'7','13','17','19','21','X'},
                 chr_column='chr',
                 batch_size=32,
                 padded_seq_len=600, 
                 num_workers=8,
                 normalize=False,
                 **kwargs):       
        """
        Takes a .txt file with a column cotaining DNA sequences,
        column(s) containing log2FC, and a chromosome column.
        Preprocesses, tokenizes, creates Train/Val/Test dataloaders.

        Parameters
        ----------
        datafile_path : TYPE
            Path to the .txt file with the data (space-separated).
        sequence_column : TYPE, optional
             Name of the column of the DNA sequences. The default is 'nt_sequence'.
        activity_columns : TYPE, optional
            List of names of the columns of the associated activity. The default is ['K562', 'HepG2', 'SKNSH'].
        chr_column : TYPE, optional
             Name of the column of the chromosome number. The default is 'chr'.
        batch_size : TYPE, optional
            Number of examples in each mini batch. The default is 32.
        padded_seq_len : TYPE, optional
            Desired total sequence length after padding. The default is 600.
        num_workers : TYPE, optional
            number of gpus(?) or cpu cores to be used. The default is 8.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        self.datafile_path = datafile_path
        self.data_project = data_project
        self.project_column = project_column
        self.sequence_column = sequence_column
        self.activity_columns = activity_columns
        self.valtest_chrs = valtest_chrs
        self.chr_column = chr_column
        self.batch_size = batch_size
        self.padded_seq_len = padded_seq_len        
        self.num_workers = num_workers
        self.normalize = normalize
        
        self.pad_column_name = 'padded_seq'
        self.tensor_column_name = 'onehot_seq'
        
        self.padding_fn = partial(utils.row_pad_sequence,
                                  in_column_name=self.sequence_column,
                                  padded_seq_len=self.padded_seq_len
                                  )
        self.tokenize_fn = partial(utils.row_dna2tensor,
                                   in_column_name=self.pad_column_name
                                   )
                
    def setup(self, stage='train'):
        #--------- parse data from MPRA file ---------
        temp_df = utils.parse_file(file_path=self.datafile_path,
                         sequence_column=self.sequence_column,
                         activity_columns=self.activity_columns,
                         chromosome_column=self.chr_column,
                         project_column=self.project_column)
        temp_df = temp_df[temp_df[self.project_column].isin(self.data_project)].reset_index(drop=True)
        
        #--------- standard score normalization ---------
        if self.normalize:
            temp_df2 = temp_df[self.activity_columns]
            self.activity_means = torch.Tensor(temp_df2.mean().to_numpy())
            self.activity_stds = torch.Tensor(temp_df2.std().to_numpy())
            temp_df[self.activity_columns] = (temp_df2 - temp_df2.mean()) / temp_df2.std()
            
        #--------- pad sequences, convert to one-hots ---------
        print('Padding sequences...')
        temp_df[self.pad_column_name] = temp_df.apply(self.padding_fn, axis=1)
        print('Tokenizing sequences...')
        temp_df[self.tensor_column_name] = temp_df.apply(self.tokenize_fn, axis=1)
        
        #--------- split dataset in train/val/test sets ---------
        print('Creating train/val/test datasets...')
        all_chrs = set(temp_df[self.chr_column])
        self.train_chrs = all_chrs - self.valtest_chrs
        
        sequences_train = list(temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.tensor_column_name])
        activities_train = temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.activity_columns].to_numpy()
        sequences_valtest = list(temp_df[temp_df[self.chr_column].isin(self.valtest_chrs)][self.tensor_column_name])
        activities_valtest = temp_df[temp_df[self.chr_column].isin(self.valtest_chrs)][self.activity_columns].to_numpy()
        
        self.data_df = temp_df
        
        self.sequences_train = torch.stack(sequences_train)
        self.activities_train = torch.Tensor(activities_train)
        self.sequences_valtest = torch.stack(sequences_valtest)
        self.activities_valtest = torch.Tensor(activities_valtest)
        
        self.dataset_train = TensorDataset(self.sequences_train, self.activities_train)
        dataset_valtest = TensorDataset(self.sequences_valtest, self.activities_valtest)
        
        self.num_examples = len(temp_df)
        self.val_size     = self.sequences_valtest.shape[0] // 2
        self.test_size    = self.sequences_valtest.shape[0] - self.val_size
        self.train_size   = self.num_examples - self.test_size - self.val_size
        
        self.dataset_val, self.dataset_test = random_split(dataset_valtest, 
                                                           [self.val_size, self.test_size],
                                                           generator=torch.Generator().manual_seed(1))
        
        train_pct = round(100 * self.train_size / self.num_examples, 2)
        val_pct   = round(100 * self.val_size / self.num_examples, 2)
        test_pct  = round(100 * self.test_size / self.num_examples, 2)
        print(f'{self.num_examples} training examples; {train_pct}%|{val_pct}%|{test_pct}% train|val|test') 
        
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

