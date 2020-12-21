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

sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')
import boda
from boda.common import constants           

#------------------------- HELPER FUNCTIONS ------------------------------------

def pad_sequence(sequence, paddedSeqLen, upStreamSeq, downStreamSeq):
    origSeqLen = len(sequence)
    paddingLen = paddedSeqLen - origSeqLen
    assert paddingLen <= (len(upStreamSeq) + len(downStreamSeq)), 'Not enough padding available'
    upPad = upStreamSeq[-paddingLen//2 + paddingLen%2:]
    downPad = downStreamSeq[:paddingLen//2 + paddingLen%2]
    paddedSequence = upPad + sequence + downPad            
    return paddedSequence

def dna2tensor(sequence, vocab):
    seqTensor = torch.zeros((len(sequence), len(vocab)))
    for letterIdx, letter in enumerate(sequence):
        seqTensor[letterIdx, vocab.index(letter)] = 1
    return seqTensor   


#------------------------- PUBLIC CLASS ----------------------------------------

class ExampleLightingData(pl.LightningDataModule):
    
    def __init__(self, file_seqID: str = './', file_seqFunc: str = './', 
                 ValSize_pct=5, TestSize_pct=5,
                 batchSize=32,
                 paddedSeqLen=600):       
        super().__init__()
        self.data_name  = 'ExampleLightingData'
        self.file_seqID = file_seqID
        self.file_seqFunc = file_seqFunc
        self.ValSize_pct = ValSize_pct
        self.TestSize_pct = TestSize_pct
        self.batchSize = batchSize
        self.paddedSeqLen = paddedSeqLen        
        
    def parse_MPRAfiles(self):
        fasta_dict = {}
        data = []
        with open(self.file_seqID, 'r') as f:           
            for line in f:
                if '>' == line[0]:
                    my_id = line.rstrip()[1:]
                    fasta_dict[my_id] = ''
                else:
                    fasta_dict[my_id] += line.rstrip()                            
        with open(self.file_seqFunc, 'r') as f:
            header = f.readline().rstrip()
            activity_idx = header.split().index('log2FoldChange')
            for line in f:
                entry = line.rstrip().split()
                try:
                    data.append( [fasta_dict[ entry[0] ], float(entry[activity_idx])] )
                except KeyError:
                    print(f'Could not find key: {entry[0]}')
                except ValueError:
                    print(f'For key: {entry[0]}, cannot convert value: {entry[activity_idx]}')
                    data.append( [fasta_dict[ entry[0] ], np.nan] )                        
        return data
                
    def prepare_data(self):        
        raise NotImplementedError
             
    def setup(self):
        #------ parse data from original MPRA files ------
        data = self.parse_MPRAfiles()
        self.num_examples = len(data)
        
        #------ pad dna sequences and convert them to one-hot tensors ------
        print('Padding sequences and converting to one-hot tensors...')
        seqTensors = []
        activities = []
        for idx,(sequence, activity) in enumerate(data):
            paddedSeq = pad_sequence(sequence, self.paddedSeqLen, constants.MPRA_UPSTREAM, constants.MPRA_DOWNSTREAM)
            seqTensor = dna2tensor(paddedSeq, vocab=constants.STANDARD_NT)
            seqTensors.append(seqTensor)
            activities.append(activity)
            if (idx+1)%5000 == 0:
                print(f'{idx+1}/{len(data)} sequences padded and one-hotted...')                                         
        sequencesTensor = torch.stack(seqTensors)
        activitiesTensor = torch.Tensor(activities)        
        self.dataset_full = TensorDataset(sequencesTensor, activitiesTensor)  
        
        #------ split dataset in train/val/test sets ------
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

    
    
#---------------------------- EXAMPLE --------------------------------------------------

# DataModule = ExampleLightingData('CMS_MRPA_092018_60K.balanced.collapsed.seqOnly.fa', 
#                                   'CMS_example_summit_shift_SKNSH_20201013.out')
# DataModule.setup()
# TrainDataloader = DataModule.train_dataloader()
# ValDataloader = DataModule.val_dataloader()
# TestDataloader = DataModule.test_dataloader()