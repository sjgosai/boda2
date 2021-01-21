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


class FastSeqProp(nn.Module):
    def __init__(self,
                 num_sequences=1,
                 seq_len=200, 
                 padding_len=400,
                 upPad_DNA=None,
                 downPad_DNA=None,
                 vocab_list=None,
                 **kwargs):
        super(FastSeqProp, self).__init__()
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
        #scaled softmax relaxation
        normalized_sequences = F.instance_norm(self.trainable_sequences)
        scaled_sequences = normalized_sequences * self.scaleWeights + self.shiftWeights
        softmaxed_sequences = F.softmax(scaled_sequences, dim=1)
        #save attributes without messing the backward graph
        self.softmaxed_sequences = softmaxed_sequences
        self.padded_softmaxed_sequences = self.pad(softmaxed_sequences)
        #sample
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
        assert self.padding_len >= 0 and type(self.padding_len) == int, 'Padding must be a nonnegative integer'
        if self.padding_len > 0:
            assert self.padding_len <= (len(self.upPad_DNA) + len(self.downPad_DNA)), 'Not enough padding available'
            upPadTensor, downPadTensor = self.dna2tensor(self.upPad_DNA), \
                                         self.dna2tensor(self.downPad_DNA)
            upPadTensor, downPadTensor = upPadTensor[:,-self.padding_len//2 + self.padding_len%2:], \
                                         downPadTensor[:,:self.padding_len//2 + self.padding_len%2]
            upPadTensor, downPadTensor = upPadTensor.repeat(self.num_sequences, 1, 1), \
                                         downPadTensor.repeat(self.num_sequences, 1, 1)
            self.register_buffer('upPadTensor', upPadTensor)
            self.register_buffer('downPadTensor', downPadTensor)
        else:
            self.upPadTensor, self.downPadTensor = None, None

        
    def pad(self, tensor):
        if self.padding_len > 0:
            paddedTensor = torch.cat([ self.upPadTensor, tensor, self.downPadTensor], dim=2)
            return paddedTensor
        else: 
            return tensor
    
    def dna2tensor(self, sequence_str):
        seqTensor = np.zeros((len(self.vocab_list), len(sequence_str)))
        for letterIdx, letter in enumerate(sequence_str):
            seqTensor[self.vocab_list.index(letter), letterIdx] = 1
        seqTensor = torch.Tensor(seqTensor)
        return seqTensor
    
    def optimize(self, predictor, loss_fn, steps=20, learning_rate=0.5, step_print=5, lr_scheduler=True):
        if lr_scheduler: etaMin = 0.000001
        else: etaMin = learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=etaMin)
        
        print('-----Initial sequence(s)-----')
        print(self.pad(self.trainable_sequences).detach().numpy())
        print('-----Training steps-----')
        loss_hist  = []
        for step in range(1, steps+1):
            optimizer.zero_grad()
            softmaxed_sequences, sampled_sequences = self()
            predictions = predictor(sampled_sequences)
            loss = loss_fn(predictions)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_hist.append(loss.item())
            if step % step_print == 0:
                print(f'step: {step}, loss: {round(loss.item(),6)}, learning_rate: {scheduler.get_last_lr()}')
        
        print('-----Final distribution-----')
        print(self.padded_softmaxed_sequences.detach().numpy())

        self.loss_hist = loss_hist
        plt.plot(loss_hist)
        plt.xlabel('Steps')
        vert_label=plt.ylabel('Loss')
        vert_label.set_rotation(90)
        plt.show()
        return None
    
    def generate(self, padded=True):
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
            return sampled_sequences.detach()



#--------------------------- EXAMPLE ----------------------------------------
if __name__ == '__main__':
    from FastSeqProp_utils import first_token_rewarder, neg_reward_loss
    import sys
    sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
    from boda.common import constants  
    
    #np.random.seed(1)                   # anchor the initial DNA sequence(s)
    #torch.manual_seed(1)                # anchor the sampling
    np.set_printoptions(precision=2)    # for shorter display of np arrays
    
    model = FastSeqProp(num_sequences=1,
                        seq_len=5,
                        padding_len=2,
                        upPad_DNA=constants.MPRA_UPSTREAM,
                        downPad_DNA=constants.MPRA_DOWNSTREAM,
                        vocab_list=constants.STANDARD_NT)
    model.optimize(predictor=first_token_rewarder,
                   loss_fn=neg_reward_loss,
                   steps=12,
                   learning_rate=0.5,
                   step_print=2,
                   lr_scheduler=True)
    sample_example = model.generate()
    
    print('-----Sample example-----')
    print(sample_example.numpy())