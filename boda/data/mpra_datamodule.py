import sys
import argparse
import tempfile
from functools import partial

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

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
        group.add_argument('--activity_columns', type=str, nargs='+', default=['K562_mean', 'HepG2_mean', 'SKNSH_mean'])
        group.add_argument('--exclude_chr_train', type=str, nargs='+', default=[''])
        group.add_argument('--val_chrs', type=str, nargs='+', default=['17','19','21','X'])
        group.add_argument('--test_chrs', type=str, nargs='+', default=['7','13'])
        group.add_argument('--chr_column', type=str, default='chr')
        group.add_argument('--std_multiple_cut', type=float, default=6.0)
        group.add_argument('--up_cutoff_move', type=float, default=3.0)
        group.add_argument('--synth_chr', type=str, default='synth')
        group.add_argument('--synth_val_pct', type=float, default=10.0)
        group.add_argument('--synth_test_pct', type=float, default=10.0)
        group.add_argument('--synth_seed', type=int, default=0)
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
                 exclude_chr_train = [''],
                 val_chrs=['17','19','21','X'],
                 test_chrs=['7','13'],
                 chr_column='chr',
                 std_multiple_cut=6.0,
                 up_cutoff_move=3.0,
                 synth_chr='synth',
                 synth_val_pct=10.0,
                 synth_test_pct=10.0,
                 synth_seed=0,
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
        datafile_path : str
            Path to the .txt file with the data (space-separated)..
        data_project : str, optional
            DESCRIPTION. The default is ['BODA', 'UKBB'].
        project_column : str, optional
            DESCRIPTION. The default is 'data_project'.
        sequence_column : str, optional
            Name of the column of the DNA sequences. The default is 'nt_sequence'.
        activity_columns : list, optional
            List of names of the columns with log2FC. The default is ['K562_mean', 'HepG2_mean', 'SKNSH_mean'].
        exclude_chr_train : list, optional
            List of chromosomes to be excluded from train. The default is [''].
        val_chrs : list, optional
            DESCRIPTION. The default is ['17','19','21','X'].
        test_chrs : list, optional
            DESCRIPTION. The default is ['7','13'].
        chr_column : str, optional
            Name of the column of the chromosome number. The default is 'chr'.
        std_multiple_cut : float, optional
            DESCRIPTION. The default is 6.0.
        up_cutoff_move : float, optional
            DESCRIPTION. The default is 3.0.
        synth_chr : str, optional
            DESCRIPTION. The default is 'synth'.
        synth_val_pct : float, optional
            DESCRIPTION. The default is 10.0.
        synth_test_pct : float, optional
            DESCRIPTION. The default is 10.0.
        synth_seed : int, optional
            DESCRIPTION. The default is 0.
        batch_size : int, optional
            Number of examples in each mini batch. The default is 32.
        padded_seq_len : int, optional
            Desired total sequence length after padding. The default is 600.
        num_workers : int, optional
            number of gpus or cpu cores to be used, right?. The default is 8.
        normalize : bool, optional
            DESCRIPTION. The default is False.
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
        self.exclude_chr_train = set(exclude_chr_train)
        self.val_chrs = set(val_chrs)
        self.test_chrs = set(test_chrs)
        self.chr_column = chr_column
        self.std_multiple_cut = std_multiple_cut
        self.up_cutoff_move = up_cutoff_move
        self.synth_chr = synth_chr
        self.synth_val_pct = synth_val_pct
        self.synth_test_pct = synth_test_pct
        self.synth_seed = synth_seed
        self.batch_size = batch_size
        self.padded_seq_len = padded_seq_len        
        self.num_workers = num_workers
        self.normalize = normalize
        
        self.pad_column_name = 'padded_seq'
        self.tensor_column_name = 'onehot_seq'
        self.activity_means = None
        self.activity_stds = None
        self.synth_chr_as_set = {synth_chr}
        
        self.padding_fn = partial(utils.row_pad_sequence,
                                  in_column_name=self.sequence_column,
                                  padded_seq_len=self.padded_seq_len
                                  )
        self.tokenize_fn = partial(utils.row_dna2tensor,
                                   in_column_name=self.pad_column_name
                                   )
                
    def setup(self, stage='train'):
        #--------- parse data from MPRA file ---------
        columns = [self.sequence_column, *self.activity_columns, self.chr_column, self.project_column]
        temp_df = utils.parse_file(file_path=self.datafile_path, columns=columns)

        temp_df = temp_df[temp_df[self.project_column].isin(self.data_project)].reset_index(drop=True)
        
        #--------- cut-off and standard score norm ---------
        means = temp_df[self.activity_columns].mean().to_numpy()
        stds  = temp_df[self.activity_columns].std().to_numpy()
        
        up_cut   = means + stds * self.std_multiple_cut + self.up_cutoff_move
        down_cut = means - stds * self.std_multiple_cut 
        
        non_extremes_filter_up = (temp_df[self.activity_columns] < up_cut).to_numpy().all(axis=1)
        temp_df = temp_df.loc[non_extremes_filter_up]
        
        non_extremes_filter_down = (temp_df[self.activity_columns] > down_cut).to_numpy().all(axis=1)
        temp_df = temp_df.loc[non_extremes_filter_down]
        
        self.num_examples = len(temp_df)
        if self.normalize:   
            temp_df[self.activity_columns] = (temp_df[self.activity_columns] - means) / stds
            self.activity_means = torch.Tensor(means)
            self.activity_stds = torch.Tensor(stds)        
        
        #--------- print cut-off info ---------
        print('-'*50)
        print('')
        for idx, cell in enumerate(self.activity_columns):
            cell_name = cell.rstrip('_mean')
            top_cut_value = round(up_cut[idx], 2)
            bottom_cut_value = round(down_cut[idx], 2)
            print(f'{cell_name} | top cut value: {top_cut_value}, bottom cut value: {bottom_cut_value}')
        print('')    
        num_up_cuts   = np.sum(~non_extremes_filter_up)
        num_down_cuts = np.sum(~non_extremes_filter_down)
        print(f'Number of examples discarded from top: {num_up_cuts}')
        print(f'Number of examples discarded from bottom: {num_down_cuts}')
        print('')
        print(f'Number of examples available: {self.num_examples}')
        print('')
        print('-'*50)
        print('')
        
        #--------- pad sequences, convert to one-hots ---------
        print('Padding sequences...')
        temp_df[self.pad_column_name] = temp_df.apply(self.padding_fn, axis=1)
        print('Tokenizing sequences...')
        
        temp_df[self.tensor_column_name] = temp_df.apply(self.tokenize_fn, axis=1)
        
        #--------- split dataset in train/val/test sets ---------
        print('Creating train/val/test datasets...')
        all_chrs = set(temp_df[self.chr_column])
        self.train_chrs = all_chrs - self.val_chrs - self.test_chrs - self.synth_chr_as_set - self.exclude_chr_train
        
        sequences_train  = list(temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.tensor_column_name])
        sequences_val    = list(temp_df[temp_df[self.chr_column].isin(self.val_chrs)][self.tensor_column_name])
        sequences_test   = list(temp_df[temp_df[self.chr_column].isin(self.test_chrs)][self.tensor_column_name])          
        activities_train = temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.activity_columns].to_numpy()
        activities_val   = temp_df[temp_df[self.chr_column].isin(self.val_chrs)][self.activity_columns].to_numpy() 
        activities_test  = temp_df[temp_df[self.chr_column].isin(self.test_chrs)][self.activity_columns].to_numpy()
            
        sequences_train  = torch.stack(sequences_train)
        sequences_val    = torch.stack(sequences_val)
        sequences_test   = torch.stack(sequences_test)        
        activities_train = torch.Tensor(activities_train)     
        activities_val   = torch.Tensor(activities_val)     
        activities_test  = torch.Tensor(activities_test)

        self.dataset_train = TensorDataset(sequences_train, activities_train)
        self.dataset_val   = TensorDataset(sequences_val, activities_val)
        self.dataset_test  = TensorDataset(sequences_test, activities_test)
             
        if self.synth_chr in all_chrs:
            synth_sequences  = list(temp_df[temp_df[self.chr_column].isin(self.synth_chr_as_set)][self.tensor_column_name])
            synth_activities = temp_df[temp_df[self.chr_column].isin(self.synth_chr_as_set)][self.activity_columns].to_numpy()
            synth_sequences  = torch.stack(synth_sequences)
            synth_activities = torch.Tensor(synth_activities)
            synth_dataset = TensorDataset(synth_sequences, synth_activities)
        
            synth_num_examples = synth_activities.shape[0]
            synth_val_size     = int(synth_num_examples * self.synth_val_pct // 100)
            synth_test_size    = int(synth_num_examples * self.synth_test_pct // 100)
            synth_train_size   = synth_num_examples - synth_val_size - synth_test_size  
    
            synth_dataset_split = random_split(synth_dataset,
                                               [synth_train_size, synth_val_size, synth_test_size],
                                               generator=torch.Generator().manual_seed(self.synth_seed))       
            self.synth_dataset_train, self.synth_dataset_val, self.synth_dataset_test = synth_dataset_split
            
            if self.synth_chr not in self.exclude_chr_train:
                self.dataset_train = ConcatDataset([self.dataset_train, self.synth_dataset_train])
            self.dataset_val   = ConcatDataset([self.dataset_val, self.synth_dataset_val])
            self.dataset_test  = ConcatDataset([self.dataset_test, self.synth_dataset_test])
        
        #--------- print train/val/test info ---------
        self.train_size   = len(self.dataset_train)
        self.val_size     = len(self.dataset_val)
        self.test_size    = len(self.dataset_test)
        train_pct = round(100 * self.train_size / self.num_examples, 2)
        val_pct   = round(100 * self.val_size / self.num_examples, 2)
        test_pct  = round(100 * self.test_size / self.num_examples, 2)
        excluded_size = self.num_examples - self.train_size - self.val_size - self.test_size
        excluded_pct = round(100 * excluded_size / self.num_examples, 2)
        print('-'*50)
        print('')
        print(f'Number of examples in train: {self.train_size} ({train_pct}%)')
        print(f'Number of examples in val:   {self.val_size} ({val_pct}%)')
        print(f'Number of examples in test:  {self.test_size} ({test_pct}%)')
        print('')
        print(f'Excluded from train: {excluded_size} ({excluded_pct})%')
        print('-'*50)       
                
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
    
    def synth_train_dataloader(self):
        return DataLoader(self.synth_dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def synth_val_dataloader(self):
        return DataLoader(self.synth_dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def synth_test_dataloader(self):
        return DataLoader(self.synth_dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)