import sys
import argparse
import tempfile
from functools import partial

import numpy as np
import pandas as pd

import torch
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset, Dataset

from ..common import constants, utils

class DNAActivityDataset(Dataset):
    """
    A PyTorch Dataset representing a collection of DNA sequences along with associated activity values.
    
    Args:
        dna_tensor (torch.Tensor): A tensor containing DNA sequences represented in one-hot encoding.
        activity_tensor (torch.Tensor): A tensor containing activity values associated with DNA sequences.
        sort_tensor (torch.Tensor, optional): A tensor used for sorting the dataset based on activity values.
        duplication_cutoff (float, optional): If provided, sequences with activity values greater than or equal
            to this cutoff will be duplicated in the dataset to balance classes.
        use_reverse_complements (bool, optional): If True, each DNA sequence is followed by its reverse complement
            in the dataset, effectively doubling the dataset size.

    Attributes:
        dna_tensor (torch.Tensor): A tensor containing DNA sequences.
        activity_tensor (torch.Tensor): A tensor containing activity values.
        duplication_cutoff (float or None): The cutoff value for duplicating sequences.
        use_reverse_complements (bool): Whether reverse complements are used.
        n_examples (int): The total number of examples in the dataset.
        n_duplicated (int): The number of duplicated examples due to class balancing.

    Methods:
        __len__(): Returns the effective length of the dataset, accounting for duplication and reverse complements.
        __getitem__(idx): Retrieves the DNA sequence and activity value at the specified index,
            considering reverse complements and duplicated sequences if applicable.
    """
    
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
    """
    PyTorch Lightning DataModule for preprocessing, tokenizing, and creating Train/Val/Test dataloaders
    for an MPRA (Massively Parallel Reporter Assay) dataset, with a column cotaining DNA sequences,
    column(s) containing log2(Fold-Change), and a chromosome column.

    Args:
        datafile_path (str): Path to the .txt file containing the (space-separated) MPRA dataset.
        data_project (list, optional): List of data project names. Default is ['BODA', 'UKBB', 'GTEX'].
        project_column (str, optional): Name of the column containing data project information. Default is 'data_project'.
        sequence_column (str, optional): Name of the column containing DNA sequences. Default is 'nt_sequence'.
        activity_columns (list, optional): List of column names containing activity values. Default is ['K562_mean', 'HepG2_mean', 'SKNSH_mean'].
        exclude_chr_train (list, optional): List of chromosomes to be excluded from training. Default is [''].
        val_chrs (list, optional): List of chromosomes for validation. Default is ['19', '21', 'X'].
        test_chrs (list, optional): List of chromosomes for testing. Default is ['7', '13'].
        chr_column (str, optional): Name of the column containing chromosome numbers. Default is 'chr'.
        std_multiple_cut (float, optional): Cut-off for extreme value filtering. Default is 6.0.
        up_cutoff_move (float, optional): Cut-off shift for extreme value filtering. Default is 4.0.
        synth_chr (str, optional): Synthetic chromosome identifier. Default is 'synth'.
        synth_val_pct (float, optional): Percentage of synthetic data for validation. Default is 10.0.
        synth_test_pct (float, optional): Percentage of synthetic data for testing. Default is 10.0.
        synth_seed (int, optional): Seed for synthetic data selection. Default is 0.
        batch_size (int, optional): Number of examples in each mini batch. Default is 32.
        padded_seq_len (int, optional): Desired total sequence length after padding. Default is 600.
        num_workers (int, optional): Number of workers for data loading. Default is 8.
        normalize (bool, optional): Apply standard score normalization. Default is False.
        duplication_cutoff (float, optional): Cutoff value for duplicating sequences during training. Default is None.
        use_reverse_complements (bool, optional): Whether to use reverse complements for data augmentation. Default is False.

    Methods:
        setup(stage='train'): Preprocesses and tokenizes the dataset based on provided parameters.
        train_dataloader(): Returns a DataLoader for the training dataset.
        val_dataloader(): Returns a DataLoader for the validation dataset.
        test_dataloader(): Returns a DataLoader for the test dataset.
        synth_train_dataloader(): Returns a DataLoader for the synthetic training dataset.
        synth_val_dataloader(): Returns a DataLoader for the synthetic validation dataset.
        synth_test_dataloader(): Returns a DataLoader for the synthetic test dataset.
        chr_train_dataloader(): Returns a DataLoader for the chromosome-based training dataset.
        chr_val_dataloader(): Returns a DataLoader for the chromosome-based validation dataset.
        chr_test_dataloader(): Returns a DataLoader for the chromosome-based test dataset.
    """

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        
        group.add_argument('--datafile_path', type=str, required=True)
        group.add_argument('--data_project', nargs='+', action=utils.ExtendAction, default=['BODA','UKBB','GTEX'])
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
        group.add_argument('--left_flank', type=str, default=constants.MPRA_UPSTREAM)
        group.add_argument('--right_flank', type=str, default=constants.MPRA_DOWNSTREAM)
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
                 left_flank=constants.MPRA_UPSTREAM,
                 right_flank=constants.MPRA_DOWNSTREAM,
                 num_workers=8,
                 normalize=False,
                 duplication_cutoff=None,
                 use_reverse_complements=False,
                 **kwargs):
        """
        Initializes the MPRA_DataModule with provided parameters.
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
        self.left_flank = left_flank
        self.right_flank = right_flank
        self.num_workers = num_workers
        self.normalize = normalize
        self.duplication_cutoff = duplication_cutoff
        self.use_reverse_complements = use_reverse_complements
        
        self.pad_column_name = 'padded_seq'
        self.activity_means = None
        self.activity_stds = None
        self.synth_chr_as_set = {synth_chr}
        
        self.padding_fn = partial(utils.row_pad_sequence,
                                  in_column_name=self.sequence_column,
                                  padded_seq_len=self.padded_seq_len,
                                  upStreamSeq=self.left_flank,
                                  downStreamSeq=self.right_flank
                                  )
        self.chr_dataset_train = None
        self.chr_dataset_val = None
        self.chr_dataset_test = None
        self.synth_dataset_train = None
        self.synth_dataset_val = None
        self.synth_dataset_test = None

    def setup(self, stage = 'train'):
        """
        Preprocesses and tokenizes the dataset based on provided parameters.
        """
        columns = [self.sequence_column, *self.activity_columns, self.chr_column, self.project_column]
        temp_df = utils.parse_file(file_path=self.datafile_path, columns=columns)

        temp_df = temp_df[temp_df[self.project_column].isin(self.data_project)].reset_index(drop=True)

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

        print('Padding sequences... \n')
        temp_df[self.pad_column_name] = temp_df.apply(self.padding_fn, axis=1)

        print('Creating train/val/test datasets with tokenized sequences... \n')
        all_chrs = set(temp_df[self.chr_column])
        self.train_chrs = all_chrs - self.val_chrs - self.test_chrs - self.synth_chr_as_set - self.exclude_chr_train

        if len(self.train_chrs) > 0:
            split_temp_df = temp_df.loc[temp_df[self.chr_column].isin(self.train_chrs)]
            list_tensor_seq = []
            for index, row in split_temp_df.iterrows():
                list_tensor_seq.append(utils.row_dna2tensor(row, in_column_name=self.pad_column_name))
            activities_train = temp_df[temp_df[self.chr_column].isin(self.train_chrs)][self.activity_columns].to_numpy()
            sequences_train  = torch.stack(list_tensor_seq)
            activities_train = torch.Tensor(activities_train)    
            self.chr_dataset_train = TensorDataset(sequences_train, activities_train)
            self.chr_dataset_train = DNAActivityDataset(sequences_train, activities_train, 
                                                        sort_tensor=torch.max(activities_train, dim=-1).values, 
                                                        duplication_cutoff=self.duplication_cutoff, 
                                                        use_reverse_complements=self.use_reverse_complements)

        if len(self.val_chrs) > 0:
            split_temp_df = temp_df.loc[temp_df[self.chr_column].isin(self.val_chrs)]
            list_tensor_seq = []
            for index, row in split_temp_df.iterrows():
                list_tensor_seq.append(utils.row_dna2tensor(row, in_column_name=self.pad_column_name))
            activities_val = temp_df[temp_df[self.chr_column].isin(self.val_chrs)][self.activity_columns].to_numpy()
            sequences_val  = torch.stack(list_tensor_seq)
            activities_val = torch.Tensor(activities_val)  
            self.chr_dataset_val = TensorDataset(sequences_val, activities_val)
        
        if len(self.test_chrs) > 0:
            split_temp_df = temp_df.loc[temp_df[self.chr_column].isin(self.test_chrs)]
            list_tensor_seq = []
            for index, row in split_temp_df.iterrows():
                list_tensor_seq.append(utils.row_dna2tensor(row, in_column_name=self.pad_column_name))
            self.chr_df_test  = temp_df[temp_df[self.chr_column].isin(self.test_chrs)]
            activities_test   = temp_df[temp_df[self.chr_column].isin(self.test_chrs)][self.activity_columns].to_numpy()    
            sequences_test    = torch.stack(list_tensor_seq)        
            activities_test   = torch.Tensor(activities_test)
            self.chr_dataset_test = TensorDataset(sequences_test, activities_test)
             
        if self.synth_chr in all_chrs:
            split_temp_df = temp_df.loc[temp_df[self.chr_column].isin(self.synth_chr_as_set)]
            list_tensor_seq = []
            for index, row in split_temp_df.iterrows():
                list_tensor_seq.append(utils.row_dna2tensor(row, in_column_name=self.pad_column_name))
            synth_activities = temp_df[temp_df[self.chr_column].isin(self.synth_chr_as_set)][self.activity_columns].to_numpy()
            synth_sequences  = torch.stack(list_tensor_seq)
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
        """
        Returns a DataLoader for the training dataset.
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
    
    def synth_train_dataloader(self):
        """
        Returns a DataLoader for the synthetic training dataset.
        """
        return DataLoader(self.synth_dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def synth_val_dataloader(self):
        """
        Returns a DataLoader for the synthetic validation dataset.
        """
        return DataLoader(self.synth_dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def synth_test_dataloader(self):
        """
        Returns a DataLoader for the synthetic test dataset.
        """
        return DataLoader(self.synth_dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
    
    def chr_train_dataloader(self):
        """
        Returns a DataLoader for the chromosome-based training dataset.
        """
        return DataLoader(self.chr_dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def chr_val_dataloader(self):
        """
        Returns a DataLoader for the chromosome-based validation dataset.
        """
        return DataLoader(self.chr_dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def chr_test_dataloader(self):
        """
        Returns a DataLoader for the chromosome-based test dataset.
        """
        return DataLoader(self.chr_dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
