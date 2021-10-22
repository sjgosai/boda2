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

class DNAActivityDataset(Dataset):
    
    def __init__(self, dna_tensor, activity_tensor, sort_tensor=None, 
                 duplication_cutoff=None, use_reverse_complements=False):
        self.dna_tensor = dna_tensor
        self.activity_tensor = activity_tensor
        self.duplication_cutoff = duplication_cutoff
        self.use_reverse_complements = use_reverse_complements
        
        self.n_examples   = self.dna_tensor.shape[0]
        self.n_duplicated = 0
        
        if duplication_cutoff is not None:
            _, sort_order = torch.sort(sort_tensor, descending=True, stable=True)
            self.dna_tensor = dna_tensor[sort_order]
            self.activity_tensor = self.activity_tensor[sort_order]
            
            self.n_duplicated = (sort_tensor >= duplication_cutoff).sum().item()
        
    def __len__(self):
        dataset_len = self.dna_tensor.shape[0] + self.n_duplicated
        if self.use_reverse_complements:
            dataset_len = 2 * dataset_len
        return dataset_len
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} is out of bounds for dataset with size {len(self)}")
        if idx < 0:
            if -idx > len(self):
                raise ValueError(f"absolute value of {idx} is out of bounds for dataset with size {len(self)}")
            
        if self.use_reverse_complements:
            take_rc = idx % 2 == 1
            item_idx= (idx // 2) % self.n_examples
        else:
            take_rc = False            
            item_idx= idx % self.n_examples
            
        dna      = self.dna_tensor[item_idx]
        activity = self.activity_tensor[item_idx]

        if take_rc:
            dna = utils.reverse_complement_onehot(dna)
        
        return dna, activity

class MPRA_DataModule(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        
        group.add_argument('--datafile_path', type=str, required=True)
        group.add_argument('--data_project', type=str, nargs='+', default=['BODA', 'UKBB', 'GTEX'])
        group.add_argument('--project_column', type=str, default='data_project')
        group.add_argument('--sequence_column', type=str, default='nt_sequence')
        group.add_argument('--activity_columns', type=str, nargs='+', default=['K562_mean', 'HepG2_mean', 'SKNSH_mean'])
        group.add_argument('--exclude_chr_train', type=str, nargs='+', default=[''])
        group.add_argument('--val_chrs', type=str, nargs='+', default=['19','21','X'])
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
        group.add_argument('--duplication_cutoff', type=float, 
                           help='sequences with max activities higher then this are duplicated in training')
        group.add_argument('--use_reverse_complements', type=utils.str2bool, default=False,
                           help='Reverse complement to augment/duplicate training examples')
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
                 data_project=['BODA', 'UKBB', 'GTEX'],
                 project_column='data_project',
                 sequence_column='nt_sequence',
                 activity_columns=['K562_mean', 'HepG2_mean', 'SKNSH_mean'],
                 exclude_chr_train=[''],
                 val_chrs=['19','21','X'],
                 test_chrs=['7','13'],
                 chr_column='chr',
                 std_multiple_cut=6.0,
                 up_cutoff_move=4.0,
                 synth_chr='synth',
                 synth_val_pct=10.0,
                 synth_test_pct=10.0,
                 synth_seed=0,
                 batch_size=32,
                 padded_seq_len=600, 
                 num_workers=8,
                 normalize=False,
                 duplication_cutoff=None,
                 use_reverse_complements=False,
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
        duplication_cutoff: float, optional
            All sequences with max activity across cell types above this value will be
            duplicated during training.
        use_reverse_complements: bool, optional
            If true, reverse complements of training sequences will be added to 
            training set.
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
        self.exclude_chr_train = set(exclude_chr_train) - {''}
        self.val_chrs = set(val_chrs) - {''}
        self.test_chrs = set(test_chrs) - {''}
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
        self.duplication_cutoff = duplication_cutoff
        self.use_reverse_complements = use_reverse_complements
        
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
        self.chr_dataset_train = None
        self.chr_dataset_val = None
        self.chr_dataset_test = None
        self.synth_dataset_train = None
        self.synth_dataset_val = None
        self.synth_dataset_test = None
                
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
        
        if len(self.train_chrs) > 0:
            sequences_train  = list(temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.tensor_column_name])
            activities_train = temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.activity_columns].to_numpy()
            sequences_train  = torch.stack(sequences_train)
            activities_train = torch.Tensor(activities_train)    
            self.chr_dataset_train = TensorDataset(sequences_train, activities_train)
            self.chr_dataset_train = DNAActivityDataset(sequences_train, activities_train, 
                                                        sort_tensor=torch.max(activities_train, dim=-1).values, 
                                                        duplication_cutoff=self.duplication_cutoff, 
                                                        use_reverse_complements=self.use_reverse_complements)
        
        if len(self.val_chrs) > 0:
            sequences_val  = list(temp_df[temp_df[self.chr_column].isin(self.val_chrs)][self.tensor_column_name])
            activities_val = temp_df[temp_df[self.chr_column].isin(self.val_chrs)][self.activity_columns].to_numpy()
            sequences_val  = torch.stack(sequences_val)
            activities_val = torch.Tensor(activities_val)  
            self.chr_dataset_val = TensorDataset(sequences_val, activities_val)
        
        if len(self.test_chrs) > 0:
            sequences_test    = list(temp_df[temp_df[self.chr_column].isin(self.test_chrs)][self.tensor_column_name])                      
            activities_test   = temp_df[temp_df[self.chr_column].isin(self.test_chrs)][self.activity_columns].to_numpy()    
            sequences_test    = torch.stack(sequences_test)        
            activities_test   = torch.Tensor(activities_test)
            self.chr_dataset_test = TensorDataset(sequences_test, activities_test)
             
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
            
            # Repackage training synth
            dna, activities = list(zip(*list(self.synth_dataset_train)))
            dna = torch.stack(dna, dim=0)
            activities = torch.stack(activities, dim=0)
            self.synth_dataset_train = DNAActivityDataset(dna, activities, 
                                                          sort_tensor=torch.max(activities, dim=-1).values, 
                                                          duplication_cutoff=self.duplication_cutoff, 
                                                          use_reverse_complements=self.use_reverse_complements)

            
            if self.chr_dataset_train is None:
                if self.synth_chr not in self.exclude_chr_train:
                    self.dataset_train = self.synth_dataset_train
            else:
                self.dataset_train = ConcatDataset([self.chr_dataset_train, self.synth_dataset_train])
            if self.chr_dataset_val is None:
                self.dataset_val = self.synth_dataset_val
            else:
                self.dataset_val = ConcatDataset([self.chr_dataset_val, self.synth_dataset_val])
            if self.chr_dataset_test is None:
                self.dataset_test = self.synth_dataset_test
            else:
                self.dataset_test = ConcatDataset([self.chr_dataset_test, self.synth_dataset_test])
        else:
            self.dataset_train = self.chr_dataset_train
            self.dataset_val = self.chr_dataset_val
            self.dataset_test = self.chr_dataset_test
        
        #--------- print train/val/test info ---------
        if self.dataset_train is not None: self.train_size = len(self.dataset_train)
        else: self.train_size = 0
            
        if self.dataset_val is not None: self.val_size = len(self.dataset_val)
        else: self.val_size = 0
            
        if self.dataset_test is not None: self.test_size = len(self.dataset_test)
        else: self.test_size = 0
            
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
    
    def chr_train_dataloader(self):
        return DataLoader(self.chr_dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def chr_val_dataloader(self):
        return DataLoader(self.chr_dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def chr_test_dataloader(self):
        return DataLoader(self.chr_dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)