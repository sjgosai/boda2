import sys
import argparse
import tempfile

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset

from ..common import constants     


class BODA2_DataModule(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        
        group.add_argument('--datafile_path', type=str, required=True)
        group.add_argument('--sequence_column', type=str, default='nt.sequence')
        group.add_argument('--activity_columns', type=str, nargs='+', default=['K562', 'HepG2.neon', 'SKNSH'])
        group.add_argument('--valid_pct', type=float, default=5, 
                           help='Percentage of examples to form the validation set')    
        group.add_argument('--test_pct', type=float, default=5, 
                           help='Percentage of examples to form the test set')          
        group.add_argument('--batch_size', type=int, default=32, 
                           help='Number of examples in each mini batch')         
        group.add_argument('--padded_seq_len', type=int, default=600, 
                           help='Desired total sequence length after padding') 
        group.add_argument('--num_workers', type=int, default=8, 
                           help='number of gpus or cpu cores to be used') 
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
                 sequence_column='nt.sequence',
                 activity_columns=['K562', 'HepG2', 'SKNSH'],
                 valid_pct=5,
                 test_pct=5,
                 batch_size=32,
                 padded_seq_len=600, 
                 num_workers=8,
                 **kwargs):       
        """
        Takes a .txt file with a column cotaining DNA sequences
        and another column containing some activity.
        Preprocesses, tokenizes, creates Train/Val/Test dataloaders.

        Parameters
        ----------
        datafile_path : TYPE
            Path to the .txt file with the data (space-separated).
        sequence_column : TYPE, optional
             Name of the column of the DNA sequences. The default is 'nt.sequence'.
        activity_columns : TYPE, optional
            List of names of the columns of the associated activity. The default is ['K562', 'HepG2', 'SKNSH'].
        valid_pct : TYPE, optional
            Percentage of examples to form the validation set. The default is 5.
        test_pct : TYPE, optional
            Percentage of examples to form the test set. The default is 5.
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
        self.sequence_column = sequence_column
        self.activity_columns = activity_columns
        self.valid_pct = valid_pct
        self.test_pct = test_pct
        self.batch_size = batch_size
        self.padded_seq_len = padded_seq_len        
        self.num_workers = num_workers
                
    def setup(self, stage='train'):             
        #--------- parse data from original MPRA files ---------
        self.raw_data = self.parse_textFile(self.datafile_path, self.sequence_column, self.activity_columns)
        self.num_examples = len(self.raw_data)
        
        #--------- pad dna sequences, convert to one-hots, create tensors ---------          
        print('Padding sequences and converting to one-hot tensors...')
        seqTensors = []
        activities = []
        for idx, data in enumerate(self.raw_data):
            sequence = data[0]
            activity = data[1:]
            paddedSeq = self.pad_sequence(sequence, self.padded_seq_len, constants.MPRA_UPSTREAM, constants.MPRA_DOWNSTREAM)
            seqTensor = self.dna2tensor(paddedSeq, vocab=constants.STANDARD_NT)
            seqTensors.append(seqTensor)
            activities.append(activity)
            if (idx+1)%10000 == 0:
                print(f'{idx+1}/{self.num_examples} sequences padded and tokenized...')                                         
        self.sequencesTensor = torch.stack(seqTensors)
        self.activitiesTensor = torch.Tensor(activities).view(-1, len(self.activity_columns))            
        self.dataset_full = TensorDataset(self.sequencesTensor, self.activitiesTensor)  
        
        #--------- split dataset in train/val/test sets ---------     
        self.val_size = int(self.num_examples * self.valid_pct // 100)      #might need to pre-separate examples in future data
        self.test_size = int(self.num_examples * self.test_pct // 100)
        self.train_size = int(self.num_examples - self.val_size - self.test_size)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.dataset_full, 
                                                                               [self.train_size, self.val_size, self.test_size],
                                                                               generator=torch.Generator().manual_seed(1))
           
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
       
    #------------------------------ HELPER METHODS ------------------------------ 
    @staticmethod
    def parse_textFile(file_path, sequence_column, activity_columns):
        df = pd.read_csv(file_path, sep=" ")
        sub_df = df[[sequence_column, *activity_columns]].dropna()
        data_list = sub_df.values.tolist()
        return data_list
    
    @staticmethod
    def pad_sequence(sequence, padded_seq_len, upStreamSeq, downStreamSeq):
        origSeqLen = len(sequence)
        paddingLen = padded_seq_len - origSeqLen
        assert paddingLen <= (len(upStreamSeq) + len(downStreamSeq)), 'Not enough padding available'
        upPad = upStreamSeq[-paddingLen//2 + paddingLen%2:]
        downPad = downStreamSeq[:paddingLen//2 + paddingLen%2]
        paddedSequence = upPad + sequence + downPad            
        return paddedSequence
    
    @staticmethod
    def dna2tensor(sequence, vocab):
        seqTensor = np.zeros((len(vocab), len(sequence)))
        for letterIdx, letter in enumerate(sequence):
            seqTensor[vocab.index(letter), letterIdx] = 1
        seqTensor = torch.Tensor(seqTensor)
        return seqTensor 
