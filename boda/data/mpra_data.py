import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
import sys

from ..common import constants     


class BODA2_DataModule(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        
        group.add_argument('--dataFile_path', type=str, required=True)
        group.add_argument('--sequenceColumn', type=str, default='nt.sequence')
        group.add_argument('--activityColumns', type=str, nargs='+', default=['K562', 'HepG2', 'SKNSH'])
        group.add_argument('--ValSize_pct', type=float, default=5, 
                           help='Percentage of examples to form the validation set')    
        group.add_argument('--TestSize_pct', type=float, default=5, 
                           help='Percentage of examples to form the test set')          
        group.add_argument('--batchSize', type=int, default=32, 
                           help='Number of examples in each mini batch')         
        group.add_argument('--paddedSeqLen', type=int, default=600, 
                           help='Desired total sequence length after padding') 
        group.add_argument('--numWorkers', type=int, default=8, 
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
                 dataFile_path,
                 sequenceColumn='nt.sequence',
                 activityColumns=['K562', 'HepG2', 'SKNSH'],
                 ValSize_pct=5,
                 TestSize_pct=5,
                 batchSize=32,
                 paddedSeqLen=600, 
                 numWorkers=8,
                 **kwargs):       
        """
        Takes a .txt file with a column cotaining DNA sequences
        and another column containing some activity.
        Preprocesses, tokenizes, creates Train/Val/Test dataloaders.

        Parameters
        ----------
        dataFile_path : TYPE
            Path to the .txt file with the data (space-separated).
        sequenceColumn : TYPE, optional
             Name of the column of the DNA sequences. The default is 'nt.sequence'.
        activityColumns : TYPE, optional
            List of names of the columns of the associated activity. The default is ['K562', 'HepG2', 'SKNSH'].
        ValSize_pct : TYPE, optional
            Percentage of examples to form the validation set. The default is 5.
        TestSize_pct : TYPE, optional
            Percentage of examples to form the test set. The default is 5.
        batchSize : TYPE, optional
            Number of examples in each mini batch. The default is 32.
        paddedSeqLen : TYPE, optional
            Desired total sequence length after padding. The default is 600.
        numWorkers : TYPE, optional
            number of gpus(?) or cpu cores to be used. The default is 8.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        self.dataFile_path = dataFile_path
        self.sequenceColumn = sequenceColumn
        self.activityColumns = activityColumns
        self.ValSize_pct = ValSize_pct
        self.TestSize_pct = TestSize_pct
        self.batchSize = batchSize
        self.paddedSeqLen = paddedSeqLen        
        self.numWorkers = numWorkers
                
    def setup(self, stage='train'):             
        #--------- parse data from original MPRA files ---------
        self.raw_data = self.parse_textFile(self.dataFile_path, self.sequenceColumn, self.activityColumns)
        self.num_examples = len(self.raw_data)
        
        #--------- pad dna sequences, convert to one-hots, create tensors ---------          
        print('Padding sequences and converting to one-hot tensors...')
        seqTensors = []
        activities = []
        for idx, data in enumerate(self.raw_data):
            sequence = data[0]
            activity = data[1:]
            paddedSeq = self.pad_sequence(sequence, self.paddedSeqLen, constants.MPRA_UPSTREAM, constants.MPRA_DOWNSTREAM)
            seqTensor = self.dna2tensor(paddedSeq, vocab=constants.STANDARD_NT)
            seqTensors.append(seqTensor)
            activities.append(activity)
            if (idx+1)%10000 == 0:
                print(f'{idx+1}/{self.num_examples} sequences padded and tokenized...')                                         
        self.sequencesTensor = torch.stack(seqTensors)
        self.activitiesTensor = torch.Tensor(activities).view(-1, len(self.activityColumns))            
        self.dataset_full = TensorDataset(self.sequencesTensor, self.activitiesTensor)  
        
        #--------- split dataset in train/val/test sets ---------     
        self.val_size = int(self.num_examples * self.ValSize_pct // 100)      #might need to pre-separate examples in future data
        self.test_size = int(self.num_examples * self.TestSize_pct // 100)
        self.train_size = int(self.num_examples - self.val_size - self.test_size)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.dataset_full, 
                                                                               [self.train_size, self.val_size, self.test_size],
                                                                               generator=torch.Generator().manual_seed(1))
           
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batchSize,
                          shuffle=True, num_workers=self.numWorkers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batchSize,
                          shuffle=False, num_workers=self.numWorkers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batchSize,
                          shuffle=False, num_workers=self.numWorkers)
       
    #------------------------------ HELPER METHODS ------------------------------ 
    @staticmethod
    def parse_textFile(file_path, sequence_column, activity_columns):
        df = pd.read_csv(file_path, sep=" ")
        sub_df = df[[sequence_column, *activity_columns]].dropna()
        data_list = sub_df.values.tolist()
        return data_list
    
    @staticmethod
    def pad_sequence(sequence, paddedSeqLen, upStreamSeq, downStreamSeq):
        origSeqLen = len(sequence)
        paddingLen = paddedSeqLen - origSeqLen
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
