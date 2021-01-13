#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:30:28 2021

@author: castrr
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

'''
'''
def create_random_DiffDNAsequences(num_sequences, seq_len):
    DNAsequences = np.zeros((num_sequences, 4, seq_len))
    for seqIdx in range(num_sequences):
        for step in range(seq_len):
            randomNucleotide = np.random.randint(4)
            DNAsequences[seqIdx, randomNucleotide, step] = 1
    DNAsequences = torch.tensor(DNAsequences, requires_grad=True)
    return DNAsequences

'''
'''
def create_scaling_weights(num_sequences):
    scaleWeights = torch.rand((numSequences, 4))
    shiftWeights = torch.rand((numSequences, 4))
    scaleWeights = scaleWeights.view(numSequences, 4, 1)
    shiftWeights = shiftWeights.view(numSequences, 4, 1)
    scaleWeights.requires_grad = True
    shiftWeights.requires_grad = True
    return scaleWeights, shiftWeights

'''
'''
def relaxation_layer(DNAsequences, scaleWeights, shiftWeights):
    normalizedSequences = F.instance_norm(DNAsequences)
    scaledSequences = normalizedSequences * scaleWeights + shiftWeights
    softmaxedSequences = F.softmax(scaledSequences, dim=1)
    return softmaxedSequences

'''
'''
def sampling_layer(softmaxedSequences):
    nucleotideDist = Categorical(torch.transpose(softmaxedSequences, 1, 2))
    sampledIdxs = nucleotideDist.sample()
    sampledSequences_T = F.one_hot(sampledIdxs, num_classes=4)        
    sampledSequences = torch.transpose(sampledSequences_T, 1, 2)
    output = sampledSequences - softmaxedSequences.detach() + softmaxedSequences   #ST estimator trick
    return output    

'''
Dummy predictor
Reward the percentage of ones in first nucleotide
'''
def first_logit_rewarder(sequences):
    weights = torch.zeros(sequences.shape)
    weights[:,0,:] = 1
    rewards = (weights * sequences).sum(2).sum(1) / sequences.shape[2]
    rewards = rewards.view(-1, 1)
    return rewards

'''
Dummy loss
For maximizing avg reward
'''
def neg_reward_loss(x):
    return -torch.mean(x)



#--------------------------- EXAMPLE ----------------------------------------
if __name__ == '__main__':
    #np.random.seed(1)                   #anchor the initial DNA sequence(s)
    #torch.manual_seed(1)                #anchor the sampling
    np.set_printoptions(precision=4)    #for shorter display of arrays

    #initialize sequence(s) and scaling parameters:
    numSequences = 1
    seqLen = 5
    DNAsequences = create_random_DiffDNAsequences(numSequences, seqLen)
    scaleWeights, shiftWeights = create_scaling_weights(numSequences)
    
    #training settings:
    LR = 1
    epochs = 10
    optimizer = torch.optim.Adam([DNAsequences, scaleWeights, shiftWeights], lr=LR, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                       T_max=epochs, 
                                                       eta_min=0.0000001, 
                                                       last_epoch=-1)
    
    print('-----Initial sequence(s)-----')
    print(DNAsequences.detach().numpy())
    loss_hist  = []
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        softmaxedSequences = relaxation_layer(DNAsequences, scaleWeights, shiftWeights)
        sampledSequences = sampling_layer(softmaxedSequences)
        predictions = first_logit_rewarder(sampledSequences)
        loss = neg_reward_loss(predictions)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())
        if epoch%2==0:
            print(f'epoch: {epoch}, loss: {round(loss.item(),6)}, learning_rate: {scheduler.get_last_lr()}')
            print('-----Updated sequence(s)-----')
            print(softmaxedSequences.detach().numpy())
    
    print('-----Last sampled sequence(s)-----')
    print(sampledSequences.detach().numpy())
    plt.plot(loss_hist)
    plt.xlabel('Gradient Steps')
    vert_label=plt.ylabel('Loss')
    vert_label.set_rotation(90)
    plt.title('Loss per epoch')
    plt.show()