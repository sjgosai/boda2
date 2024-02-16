import argparse
import sys
import random
import math
import re
import time
import os
import tempfile
import tarfile
import subprocess
import shutil

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants
from .. import model as _model

def install(package):
    """
    Install a Python package using pip.

    Args:
        package (str): Name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def set_all_seeds(seed):
    """
    Set random seeds for various libraries to ensure reproducibility.

    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def str2bool(v):
    """
    Convert a string to a boolean value.
    Pulled from https://stackoverflow.com/a/43357954

    Args:
        v (str): String to be converted.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string does not represent a valid boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
class ExtendAction(argparse.Action):
    """
    Custom argparse action to extend a list attribute.
    Pulled from https://bugs.python.org/issue16399#msg224964
    an extend action with default override

    This action allows the same option to be specified multiple times on the command line,
    extending the list of values.

    Args:
        argparse.Action: The base action class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.default, Iterable):
            self.default = [self.default]
        self.reset_dest = False
    def __call__(self, parser, namespace, values, option_string=None):
        if not self.reset_dest:
            setattr(namespace, self.dest, [])
            self.default = []
            self.reset_dest = True
        getattr(namespace, self.dest).extend(values)

        
def dna2tensor(sequence_str, vocab_list=constants.STANDARD_NT):
    """
    Convert a DNA sequence to a one-hot encoded tensor.

    Args:
        sequence_str (str): DNA sequence string.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letterIdx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letterIdx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor
    

def create_paddingTensors(num_sequences, padding_len, num_st_samples=1, for_multi_sampling=True):
    """
    Create padding tensors to append to sequences.

    Args:
        num_sequences (int): Number of sequences.
        padding_len (int): Total length of padding (maximally evenly distributed in up and down stream).
        num_st_samples (int, optional): Number of samples for stochastic tensor. Defaults to 1.
        for_multi_sampling (bool, optional): Flag for multi-sampling. Defaults to True.

    Returns:
        torch.Tensor, torch.Tensor: Padding tensors for the upstream and downstream regions.
    """
    assert padding_len >= 0 and type(padding_len) == int, 'Padding must be a nonnegative integer'
    upPad_logits, downPad_logits = None, None  
    if padding_len > 0:
        assert padding_len <= (len(constants.MPRA_UPSTREAM) + len(constants.MPRA_DOWNSTREAM)), 'Not enough padding available'
        upPad_logits, downPad_logits = dna2tensor(constants.MPRA_UPSTREAM), \
                                     dna2tensor(constants.MPRA_DOWNSTREAM)
        upPad_logits, downPad_logits = upPad_logits[:,-padding_len//2 + padding_len%2:], \
                                     downPad_logits[:,:padding_len//2 + padding_len%2]
        if for_multi_sampling:
            upPad_logits, downPad_logits = upPad_logits.repeat(num_st_samples, num_sequences, 1, 1), \
                                        downPad_logits.repeat(num_st_samples, num_sequences, 1, 1)                                     
        else:
            upPad_logits, downPad_logits = upPad_logits.repeat(num_sequences, 1, 1), \
                                         downPad_logits.repeat(num_sequences, 1, 1)  
    return upPad_logits, downPad_logits
        

def first_token_rewarder(sequences, pct=1.):
    """
    Calculate a reward based on the number of first tokens in sequences.

    Args:
        sequences (torch.Tensor): Sequences tensor.
        pct (float, optional): Desired percentage. Defaults to 1.

    Returns:
        torch.Tensor: Reward tensor.
    """
    weights = torch.zeros_like(sequences)
    weights[:,0,:] = 1
    rewards = (weights * sequences).sum(2).sum(1).div(sequences.shape[2])
    rewards = rewards.view(-1, 1)
    return 1 - abs(rewards - pct)


def neg_reward_loss(x):
    """
    Compute the negative reward loss.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Negative reward loss.
    """
    return -torch.sum(x)

def generate_all_kmers(k=4):
    """
    Generate all possible DNA k-mers.

    Args:
        k (int, optional): Length of k-mers. Defaults to 4.

    Yields:
        str: Generated k-mer.
    """
    for i in range(4**k):
        yield "".join([ constants.STANDARD_NT[ (i // (4**j)) % 4 ] for j in range(k) ])            

def organize_args(parser, args):
    """
    Organize parsed arguments into groups.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Organized arguments grouped by category.
    """
    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)
    return arg_groups

def parse_file(file_path, columns):
    """
    Parse a file and extract specified columns.

    Args:
        file_path (str): Path to the input file.
        columns (list): List of column names to extract.

    Returns:
        pandas.DataFrame: Parsed data in a DataFrame.
    """
    df = pd.read_csv(file_path, sep=" ", low_memory=False)
    sub_df = df[columns].dropna()
    return sub_df

def row_pad_sequence(row,
                     in_column_name='nt_sequence',
                     padded_seq_len=400,
                     upStreamSeq=constants.MPRA_UPSTREAM,
                     downStreamSeq=constants.MPRA_DOWNSTREAM):
    """
    Pad a sequence in a row to a specified length.

    Args:
        row (pandas.Series): Sequence row.
        in_column_name (str, optional): Name of the input column. Defaults to 'nt_sequence'.
        padded_seq_len (int, optional): Desired padded sequence length. Defaults to 400.
        upStreamSeq (str, optional): Upstream sequence. Defaults to constants.MPRA_UPSTREAM.
        downStreamSeq (str, optional): Downstream sequence. Defaults to constants.MPRA_DOWNSTREAM.

    Returns:
        str: Padded sequence.
    """

    sequence = row[in_column_name]
    origSeqLen = len(sequence)
    paddingLen = padded_seq_len - origSeqLen
    assert paddingLen <= (len(upStreamSeq) + len(downStreamSeq)), 'Not enough padding available'
    if paddingLen > 0:
        if -paddingLen//2 + paddingLen%2 < 0:
            upPad = upStreamSeq[-paddingLen//2 + paddingLen%2:]
        else:
            upPad = ''
        downPad = downStreamSeq[:paddingLen//2 + paddingLen%2]
        paddedSequence = upPad + sequence + downPad
        assert len(paddedSequence) == padded_seq_len, 'Kiubo?'
        return paddedSequence
    else:
        return sequence

def row_dna2tensor(row, in_column_name='padded_seq' , vocab=constants.STANDARD_NT):
    """
    Convert a DNA sequence row to a one-hot encoded tensor.

    Args:
        row (pandas.Series): Sequence row.
        in_column_name (str, optional): Name of the input column. Defaults to 'padded_seq'.
        vocab (list, optional): List of DNA nucleotide characters. Defaults to constants.STANDARD_NT.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    sequence_str = row[in_column_name]
    seq_idxs = torch.tensor([vocab.index(letter) for letter in sequence_str])
    sequence_tensor = F.one_hot(seq_idxs, num_classes=4).transpose(1,0)
    return sequence_tensor.type(torch.float32)

def generate_all_onehots(k=4, num_classes=4):
    """
    Generate all possible one-hot encoded k-mers.

    Args:
        k (int, optional): Length of k-mers. Defaults to 4.
        num_classes (int, optional): Number of classes. Defaults to 4.

    Returns:
        torch.Tensor: One-hot encoded tensor of all k-mers.
    """
    tokens = torch.tensor( 
        [ 
            [ 
                (i // (num_classes**j)) % num_classes 
                for j in range(k) 
            ] 
            for i in range(num_classes**k) 
        ] 
    )
    onehots = F.one_hot( tokens, num_classes=num_classes ) \
                .permute(0,2,1)
    return onehots

class KmerFilter(nn.Module):
    """
    K-mer filtering module.

    Args:
        k (int): Length of k-mers.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.register_buffer('weight', generate_all_onehots(k,4).float())
        
    def forward(self, input_):
        return F.conv1d(input_, self.weight).eq(self.k).float()

def batch2list(batch, vocab_list=constants.STANDARD_NT):
    """
    Convert a batch of one-hot encoded sequences to a list of DNA sequences.

    Args:
        batch (torch.Tensor): Batch of one-hot encoded sequences.
        vocab_list (list, optional): List of DNA nucleotide characters. Defaults to constants.STANDARD_NT.

    Returns:
        str: DNA sequence converted from one-hot encoding.
    """
    assert len(batch.shape) == 3, "Expects 3D tensor [batch, channel, length]"
    assert batch.shape[1] == len(constants.STANDARD_NT), 'Channel dim size must equal length of vocab_list'
    
    batch_size = batch.shape[0]
    idx_tensor = torch.argmax(batch, dim=1)
    for seq_idx in range(batch_size):
        idxs = idx_tensor[seq_idx, :].numpy()
        yield ''.join([ vocab_list[idx] for idx in idxs ])
    
    
def batch2fasta(batch, file_name):
    """
    Convert a batch of one-hot encoded sequences to a FASTA file.

    Args:
        batch (torch.Tensor): Batch of one-hot encoded sequences.
        file_name (str): Name of the output FASTA file.

    Writes:
        Creates a FASTA file with the batch sequences.

    Note:
        The batch should have the shape (batch_size, sequence_length, num_nucleotides).
    """
    with open(file_name, 'w') as ofile:
        batch_size = batch.shape[0]
        #seq_list = []
        for seq_idx in range(batch_size):
            seq_name = 'sequence_' + str(seq_idx)
            seq_tensor = batch[seq_idx, :, :]
            idxs = torch.argmax(seq_tensor, dim=0).numpy()
            sequence_str = ''
            for idx in idxs:
                sequence_str += constants.STANDARD_NT[idx]
            #seq_list.append(sequence_str)
            ofile.write(">" + seq_name + "\n" + sequence_str + "\n")
            
def reverse_complement_onehot(x, nt_order=constants.STANDARD_NT, 
                              complements=constants.DNA_COMPLEMENTS):
    """
    Returns the reverse complement of a onehot DNA sequence tensor.
    
    Parameters
    ----------
    x : torch.tensor
        A one-hot DNA tensor
    
    nt_order: list
        A list of nucleotide tokens with same ordering as one-hot encoding
        
    complements: dict
        A dictionary specifying complementary nucleotides, one-to-one.
        
    Returns
    -------
    torch.tensor
    """
    
    comp_alphabet = [ complements[nt] for nt in nt_order ]
    permutation = [ nt_order.index(nt) for nt in comp_alphabet ]
    
    return torch.flip(x[..., permutation, :], dims=[-1])

def align_to_alphabet(x, in_order=['A','C','G','T'], out_order=constants.STANDARD_NT):
    """
    Reorder the channel dimension of a tensor (e.g. of shape [..., C, L])
    
    Prameters
    ---------
    x: torch.tensor
        A tensor of shape [..., C, L] where the C channels are ordered 
        by their correspondence to `in_order`
        
    in_order: list
        A list that specifies the ordering of an alphabet that is consistant 
        with the input, `x`.
        
    out_order: list
        A list that specifies the target ordering of the alphabet which will 
        be used to permute the appropriate dimension of `x`.
    
    Returns
    -------
    torch.tensor
    """
    
    permutation = [ in_order.index(tk) for tk in out_order ]
    return x[..., permutation, :]

###########
# Modules #
###########

class FlankBuilder(nn.Module):
    """
    A module that adds flanking sequences to input samples.

    This module is designed to add left and right flanking sequences to input samples. The flanking sequences
    can be specified during initialization. The module can be used as a part of a neural network architecture.

    Args:
        left_flank (torch.Tensor, optional): Left flanking sequence tensor. Default is None.
        right_flank (torch.Tensor, optional): Right flanking sequence tensor. Default is None.
        batch_dim (int, optional): Batch dimension for the input sample. Default is 0.
        cat_axis (int, optional): Axis along which to concatenate the flanking sequences. Default is -1.

    Attributes:
        left_flank (torch.Tensor): Left flanking sequence tensor.
        right_flank (torch.Tensor): Right flanking sequence tensor.
        batch_dim (int): Batch dimension for the input sample.
        cat_axis (int): Axis along which to concatenate the flanking sequences.

    Methods:
        add_flanks(my_sample): Adds the specified flanking sequences to the input sample.
        forward(my_sample): Adds the flanking sequences to the input sample and returns the result.

    Example:
        left_flank = torch.zeros(1, 4, 10)  # Left flanking sequence tensor
        right_flank = torch.ones(1, 4, 5)   # Right flanking sequence tensor
        flanker = FlankBuilder(left_flank, right_flank, batch_dim=0, cat_axis=-1)
        input_sample = torch.randn(1, 4, 20)  # Input sample tensor
        output_sample = flanker(input_sample)  # Output sample with flanking sequences

    """
    def __init__(self,
                 left_flank=None,
                 right_flank=None,
                 batch_dim=0,
                 cat_axis=-1
                ):
        """
        Initialize the FlankBuilder module.

        Args:
            left_flank (torch.Tensor, optional): Left flanking sequence tensor. Default is None.
            right_flank (torch.Tensor, optional): Right flanking sequence tensor. Default is None.
            batch_dim (int, optional): Batch dimension for the input sample. Default is 0.
            cat_axis (int, optional): Axis along which to concatenate the flanking sequences. Default is -1.
        """
        super().__init__()
        
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        
        self.batch_dim = batch_dim
        self.cat_axis  = cat_axis
        
    def add_flanks(self, my_sample):
        """
        Adds the specified flanking sequences to the input sample.

        Args:
            my_sample (torch.Tensor): Input sample tensor.

        Returns:
            torch.Tensor: Output tensor with added flanking sequences.
        """
        *batch_dims, channels, length = my_sample.shape
        
        pieces = []
        
        if self.left_flank is not None:
            pieces.append( self.left_flank.expand(*batch_dims, -1, -1) )
            
        pieces.append( my_sample )
        
        if self.right_flank is not None:
            pieces.append( self.right_flank.expand(*batch_dims, -1, -1) )
            
        return torch.cat( pieces, axis=self.cat_axis )
    
    def forward(self, my_sample):
        """
        Adds the flanking sequences to the input sample and returns the result.

        Args:
            my_sample (torch.Tensor): Input sample tensor.

        Returns:
            torch.Tensor: Output tensor with added flanking sequences.
        """
        return self.add_flanks(my_sample)

######################
# PTL Module loading #
######################

def unpack_artifact(artifact_path,download_path='./'):
    """
    Unpack a tar archive artifact.

    Args:
        artifact_path (str): Path to the artifact.
        download_path (str, optional): Path to extract the artifact. Defaults to './'.
    """
    if 'gs' in artifact_path:
        subprocess.call(['gsutil','cp',artifact_path,download_path])
        if os.path.isdir(download_path):
            tar_model = os.path.join(download_path, os.path.basename(artifact_path))
        elif os.path.isfile(download_path):
            tar_model = download_path
    else:
        assert os.path.isfile(artifact_path), "Could not find file at expected path."
        tar_model = artifact_path
        
    assert tarfile.is_tarfile(tar_model), f"Expected a tarfile at {tar_model}. Not found."
    
    shutil.unpack_archive(tar_model, download_path)
    print(f'archive unpacked in {download_path}', file=sys.stderr)

def model_fn(model_dir):
    """
    Load a model from a directory.

    Args:
        model_dir (str): Path to the model directory.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    checkpoint = torch.load(os.path.join(model_dir,'torch_checkpoint.pt'))
    model_module = getattr(_model, checkpoint['model_module'])
    model        = model_module(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    return model

def load_model(artifact_path):
    
    USE_CUDA = torch.cuda.device_count() >= 1
    if os.path.isdir('./artifacts'):
        shutil.rmtree('./artifacts')

    unpack_artifact(artifact_path)

    model_dir = './artifacts'

    my_model = model_fn(model_dir)
    my_model.eval()
    if USE_CUDA:
        my_model.cuda()
    
    return my_model

'''
def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
        
def reset_weight(w):
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    
def reset_bias(b, w=None):
    if w is None:
        bound = 1 / math.sqrt(b.numel())
    else:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(b, -bound, bound)

def reshape_and_load_weights(model, state_dict):
    model_dict = model.state_dict()
    return NotImplementedError

def replace_and_load_weights(model, state_dict):
    return NotImplementedError

loaded_weights = torch.load('../../artifacts/my-model.epoch_5-step_19885.pkl')
model.load_state_dict(loaded_weights, strict=False)

pretrained_dict = ...
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(pretrained_dict)
'''