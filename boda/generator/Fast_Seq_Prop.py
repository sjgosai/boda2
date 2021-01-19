#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:22:03 2021

@author: castrr
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from warnings import warn

import sys
sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
import boda
from boda.common import constants     


class Fast_Seq_Prop(nn.Module):
    def __init__(self,
                 num_sequences=1,
                 seq_len=200, 
                 padding_len=400,
                 upPad_DNA=None,
                 downPad_DNA=None,
                 vocab_list=None,
                 **kwargs):
        super(Fast_Seq_Prop, self).__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len  
        self.padding_len = padding_len
        self.upPad_DNA = upPad_DNA
        self.downPad_DNA = downPad_DNA
        self.vocab_list = vocab_list
        self.create_paddingTensors()
        
        #initialize the trainable tensors
        self.create_random_trainable_sequences()
        self.create_scaling_weights()       
        self.softmaxed_sequences = None
        
    def forward(self):
        normalized_sequences = F.instance_norm(self.trainable_sequences)
        scaled_sequences = normalized_sequences * self.scaleWeights + self.shiftWeights
        softmaxed_sequences = F.softmax(scaled_sequences, dim=1)
        #--------------------------------------------------------
        self.softmaxed_sequences = softmaxed_sequences
        self.padded_softmaxed_sequences = self.pad(softmaxed_sequences)
        #--------------------------------------------------------
        nucleotideProbs = Categorical(torch.transpose(softmaxed_sequences, 1, 2))
        sampledIdxs = nucleotideProbs.sample()
        sampled_sequences_T = F.one_hot(sampledIdxs, num_classes=4)        
        sampled_sequences = torch.transpose(sampled_sequences_T, 1, 2)
        sampled_sequences = sampled_sequences - softmaxed_sequences.detach() + softmaxed_sequences  #ST estimator trick
        softmaxed_sequences, sampled_sequences = self.pad(softmaxed_sequences), self.pad(sampled_sequences)
        return softmaxed_sequences, sampled_sequences
    
    def create_random_trainable_sequences(self):
        trainable_sequences = np.zeros((self.num_sequences, 4, self.seq_len))
        for seqIdx in range(self.num_sequences):
            for step in range(self.seq_len):
                randomNucleotide = np.random.randint(4)
                trainable_sequences[seqIdx, randomNucleotide, step] = 1
        self.trainable_sequences = nn.Parameter(torch.tensor(trainable_sequences, dtype=torch.float))
        
    
    def create_scaling_weights(self):
        self.scaleWeights = nn.Parameter(torch.rand((self.num_sequences, 4, 1)))
        self.shiftWeights = nn.Parameter(torch.rand((self.num_sequences, 4, 1)))
        
    def create_paddingTensors(self):   
        assert self.padding_len <= (len(self.upPad_DNA) + len(self.downPad_DNA)), 'Not enough padding available'
        upPadTensor, downPadTensor = self.dna2tensor(self.upPad_DNA), \
                                     self.dna2tensor(self.downPad_DNA)
        upPadTensor, downPadTensor = upPadTensor[:,-self.padding_len//2 + self.padding_len%2:], \
                                     downPadTensor[:,:self.padding_len//2 + self.padding_len%2]
        upPadTensor, downPadTensor = upPadTensor.repeat(self.num_sequences, 1, 1), \
                                     downPadTensor.repeat(self.num_sequences, 1, 1)
        self.upPadTensor, self.downPadTensor = upPadTensor, downPadTensor
    
    def pad(self, tensor):
        paddedTensor = torch.cat([ self.upPadTensor, tensor, self.downPadTensor], dim=2)
        return paddedTensor
    
    def dna2tensor(self, sequence_str):
        seqTensor = np.zeros((len(self.vocab_list), len(sequence_str)))
        for letterIdx, letter in enumerate(sequence_str):
            seqTensor[self.vocab_list.index(letter), letterIdx] = 1
        seqTensor = torch.Tensor(seqTensor)
        return seqTensor
    
    def sample(self, padded=True):
        if self.softmaxed_sequences == None:
            warn('The model hasn\'t been trained')
            return self.pad(self.trainable_sequences)
        else:
            nucleotideProbs = Categorical(torch.transpose(self.softmaxed_sequences, 1, 2))
            sampledIdxs = nucleotideProbs.sample()
            sampled_sequences_T = F.one_hot(sampledIdxs, num_classes=4)        
            sampled_sequences = torch.transpose(sampled_sequences_T, 1, 2)
            if padded:
                sampled_sequences = self.pad(sampled_sequences)
            return sampled_sequences


#--------------------------- EXAMPLE ----------------------------------------
if __name__ == '__main__':
    from FastSeqProp_utils import first_token_rewarder, neg_reward_loss
    
    #np.random.seed(1)                   #anchor the initial DNA sequence(s)
    #torch.manual_seed(1)                #anchor the sampling
    np.set_printoptions(precision=2)    #for shorter display of np arrays
    
    numSequences = 1
    seqLen = 3
    paddingLen = 2
    LR = 1
    epochs = 10
    
    FSP_model = Fast_Seq_Prop(num_sequences=numSequences,
                              seq_len=seqLen,
                              padding_len=paddingLen,
                              upPad_DNA=constants.MPRA_UPSTREAM,
                              downPad_DNA=constants.MPRA_DOWNSTREAM,
                              vocab_list=constants.STANDARD_NT)
    optimizer = torch.optim.Adam(FSP_model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=epochs, 
                                                           eta_min=0.0000001, 
                                                           last_epoch=-1)
    
    print('-----Initial sequence(s)-----')
    print(FSP_model.pad(FSP_model.trainable_sequences).detach().numpy())   
    reward_hist  = []
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        softmaxed_sequences, sampled_sequences = FSP_model()
        predictions = first_token_rewarder(sampled_sequences)
        loss = neg_reward_loss(predictions)
        loss.backward()
        optimizer.step()
        scheduler.step()
        reward_hist.append(-loss.item())
        if epoch%2==0:
            print(f'epoch: {epoch}, reward: {round(-loss.item(),6)}') #, learning_rate: {scheduler.get_last_lr()}')
            print('-----Updated sequence(s)-----')
            print(softmaxed_sequences.detach().numpy()) 
    
    
    print('-----Last sampled sequence(s)-----')
    print(sampled_sequences.detach().numpy())
    plt.plot(reward_hist)
    plt.xlabel('Gradient Steps')
    vert_label=plt.ylabel('Reward')
    vert_label.set_rotation(90)
    plt.title('Reward per epoch')
    plt.show()