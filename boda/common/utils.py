import argparse
import torch
import numpy as np

import sys
sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
from boda.common import constants 

def set_seed(seed):
    """Fixes all random seeds
    
    Parameters
    ----------
    seed : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def str2bool(v):
    """Pulled from https://stackoverflow.com/a/43357954
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def dna2tensor(sequence_str, vocab_list=constants.STANDARD_NT):
    """
    
    Parameters
    ----------
    sequence_str : str
        A nucleotide letter sequence.
    vocab_list : list, optional
        Nucleotide vocabulary. The default is constants.STANDARD_NT.

    Returns
    -------
    seq_tensor : torch tensor
        Tokenized tensor.
        
    """
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letterIdx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letterIdx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor
    

def create_paddingTensors(num_sequences, padding_len):
    """
    
    Parameters
    ----------
    num_sequences : int
        Number of sequences that will be padded.
    padding_len : int
        Total length of padding (maximally evenly distributed in up and down stream).

    Returns
    -------
    upPad_logits : torch tensor
        Tokenized tensor of correct length from up stream DNA.
    downPad_logits : torch tensor
        Tokenized tensor of correct length from down stream DNA.
        
    """
    assert padding_len >= 0 and type(padding_len) == int, 'Padding must be a nonnegative integer'
    upPad_logits, downPad_logits = None, None  
    if padding_len > 0:
        assert padding_len <= (len(constants.MPRA_UPSTREAM) + len(constants.MPRA_DOWNSTREAM)), 'Not enough padding available'
        upPad_logits, downPad_logits = dna2tensor(constants.MPRA_UPSTREAM), \
                                     dna2tensor(constants.MPRA_DOWNSTREAM)
        upPad_logits, downPad_logits = upPad_logits[:,-padding_len//2 + padding_len%2:], \
                                     downPad_logits[:,:padding_len//2 + padding_len%2]
        upPad_logits, downPad_logits = upPad_logits.repeat(num_sequences, 1, 1), \
                                     downPad_logits.repeat(num_sequences, 1, 1)     
    return upPad_logits, downPad_logits
        

def first_token_rewarder(sequences):
    """Predictor for dummy examples

    Parameters
    ----------
    sequences : torch tensor
        Tokenized tensor.

    Returns
    -------
    rewards : 1-D tensor
        Percentage of presence of first token.

    """
    weights = torch.zeros_like(sequences)
    weights[:,0,:] = 1
    rewards = (weights * sequences).sum(2).sum(1) / sequences.shape[2]
    rewards = rewards.view(-1, 1)
    return rewards


def neg_reward_loss(x):
    """Loss for dummy examples
    For maximizing avg reward
    """
    return -torch.sum(x)