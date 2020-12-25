#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:51:51 2020

@author: castrr
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
import sys

sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
import boda
from boda.common import constants           


'''
Pytorch Lighting DataModule
Takes MPRA files, preprocesses, tokenizes, and creates Train/Val/Test dataloaders.
Arguments:
    file_seqID - Path to the file containing IDs of DNA sequences
    file_seqFunc - Path to the file containing a map of IDs to MPRA data
    MPRA_column - The name of the column of the desired activity in file_seqFunc
    ValSize_pct - percentage of examples to form the validation set
    TestSize_pct - percentage of examples to form the test set
    bathSize - number of examples in each mini batch
    paddedSeqLen - desired total sequence length after padding
'''
class MPRADataModule(pl.LightningDataModule):    
    
    def __init__(self, file_seqID: str = './', file_seqFunc: str = './',
                 MPRA_column='log2FoldChange',
                 ValSize_pct=5, TestSize_pct=5,
                 batchSize=32,
                 paddedSeqLen=600):       
        super().__init__()
        self.data_name  = 'ExampleLightingData'
        self.file_seqID = file_seqID
        self.file_seqFunc = file_seqFunc
        self.MPRA_column = MPRA_column
        self.ValSize_pct = ValSize_pct
        self.TestSize_pct = TestSize_pct
        self.batchSize = batchSize
        self.paddedSeqLen = paddedSeqLen        

    @staticmethod
    def parse_MPRAfiles(file_seqID, file_seqFunc, MPRA_column):
        fasta_dict = {}
        data = []
        with open(file_seqID, 'r') as f:           
            for line in f:
                if '>' == line[0]:
                    my_id = line.rstrip()[1:]
                    fasta_dict[my_id] = ''
                else:
                    fasta_dict[my_id] += line.rstrip()                            
        with open(file_seqFunc, 'r') as f:
            header = f.readline().rstrip()
            activity_idx = header.split().index(MPRA_column)
            for line in f:
                entry = line.rstrip().split()
                try:
                    data.append( [fasta_dict[ entry[0] ], float(entry[activity_idx])] )
                except KeyError:
                    print(f'Could not find key: {entry[0]}')
                except ValueError:
                    print(f'For key: {entry[0]}, cannot convert value: {entry[activity_idx]}')
                    #data.append( [fasta_dict[ entry[0] ], np.nan] )                        
        return data
    
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
        seqTensor = np.zeros((len(sequence), len(vocab)))
        for letterIdx, letter in enumerate(sequence):
            seqTensor[letterIdx, vocab.index(letter)] = 1
        seqTensor = torch.Tensor(seqTensor)
        return seqTensor 
    
    #------------------------------ KEY METHODS ------------------------------
          
    def prepare_data(self):        
        raise NotImplementedError
             
    def setup(self):             
        #--------- parse data from original MPRA files ---------
        self.raw_data = self.parse_MPRAfiles(self.file_seqID, self.file_seqFunc, self.MPRA_column)
        self.num_examples = len(self.raw_data)
        
        #--------- pad dna sequences, convert to one-hots, create tensors ---------          
        print('Padding sequences and converting to one-hot tensors...')
        seqTensors = []
        activities = []
        for idx,(sequence, activity) in enumerate(self.raw_data):
            paddedSeq = self.pad_sequence(sequence, self.paddedSeqLen, constants.MPRA_UPSTREAM, constants.MPRA_DOWNSTREAM)
            seqTensor = self.dna2tensor(paddedSeq, vocab=constants.STANDARD_NT)
            seqTensors.append(seqTensor)
            activities.append(activity)
            if (idx+1)%10000 == 0:
                print(f'{idx+1}/{self.num_examples} sequences padded and one-hotted...')                                         
        sequencesTensor = torch.stack(seqTensors)
        activitiesTensor = torch.Tensor(activities)        
        self.dataset_full = TensorDataset(sequencesTensor, activitiesTensor)  
        
        #--------- split dataset in train/val/test sets ---------
        self.val_size = int(np.floor(self.num_examples * self.ValSize_pct/100))      #we might need to pre-separate examples
        self.test_size = int(np.floor(self.num_examples * self.TestSize_pct/100))
        self.train_size = self.num_examples - self.val_size - self.test_size
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.dataset_full, 
                                                                               [self.train_size, self.val_size, self.test_size],
                                                                               generator=torch.Generator().manual_seed(1))
           
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batchSize)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batchSize)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batchSize)

    
    
#------------------------------- EXAMPLE --------------------------------------------------
# import time
# start_time = time.perf_counter()

# DataModule = MPRADataModule('CMS_MRPA_092018_60K.balanced.collapsed.seqOnly.fa', 
#                                   'CMS_example_summit_shift_SKNSH_20201013.out')
# DataModule.setup()
# TrainDataloader = DataModule.train_dataloader()
# ValDataloader = DataModule.val_dataloader()
# TestDataloader = DataModule.test_dataloader()

# end_time = time.perf_counter()
# run_time = end_time - start_time
# print(f"Finished in {run_time:.4f} secs")