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

def set_all_seeds(seed):
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
    random.seed(seed)
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
        
class ExtendAction(argparse.Action):
    """Pulled from https://bugs.python.org/issue16399#msg224964
       an extend action with default override
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.default, Iterable):
            self.default = [self.default]
        self.reset_dest = False
        print("------------> poised for reset", file=sys.stdout)
    def __call__(self, parser, namespace, values, option_string=None):
        if not self.reset_dest:
            print("------------> trigger reset", file=sys.stdout)
            setattr(namespace, self.dest, [])
            self.default = []
            self.reset_dest = True
        else:
            print("------------> no reset", file=sys.stdout)
        getattr(namespace, self.dest).extend(values)
        print(getattr(namespace, self.dest))

        
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
    

def create_paddingTensors(num_sequences, padding_len, num_st_samples=1, for_multi_sampling=True):
    """
    
    Parameters
    ----------
    num_sequences : int
        Number of sequences that will be padded.
    padding_len : int
        Total length of padding (maximally evenly distributed in up and down stream).
    num_st_samples: int
        Number of straight-through samples per sequence that will be padded.
        The default is 1.
    for_multi_sampling: bool
        Whether or not the tensor will pad multi-sampled sequences.
        The defaul is True.

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
        if for_multi_sampling:
            upPad_logits, downPad_logits = upPad_logits.repeat(num_st_samples, num_sequences, 1, 1), \
                                        downPad_logits.repeat(num_st_samples, num_sequences, 1, 1)                                     
        else:
            upPad_logits, downPad_logits = upPad_logits.repeat(num_sequences, 1, 1), \
                                         downPad_logits.repeat(num_sequences, 1, 1)  
    return upPad_logits, downPad_logits
        

def first_token_rewarder(sequences, pct=1.):
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
    rewards = (weights * sequences).sum(2).sum(1).div(sequences.shape[2])
    rewards = rewards.view(-1, 1)
    return 1 - abs(rewards - pct)


def neg_reward_loss(x):
    """Loss for dummy examples
    For maximizing avg reward
    """
    return -torch.sum(x)

def generate_all_kmers(k=4):
    for i in range(4**k):
        yield "".join([ constants.STANDARD_NT[ (i // (4**j)) % 4 ] for j in range(k) ])            

def organize_args(parser, args):
    arg_groups={}
    print(args)
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)
    return arg_groups

def parse_file(file_path, columns):
    df = pd.read_csv(file_path, sep=" ", low_memory=False)
    sub_df = df[columns].dropna()
    return sub_df

def row_pad_sequence(row,
                     in_column_name='nt_sequence',
                     padded_seq_len=400,
                     upStreamSeq=constants.MPRA_UPSTREAM,
                     downStreamSeq=constants.MPRA_DOWNSTREAM):
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
    sequence_str = row[in_column_name]
    seq_idxs = torch.tensor([vocab.index(letter) for letter in sequence_str])
    sequence_tensor = F.one_hot(seq_idxs, num_classes=4).transpose(1,0)
    return sequence_tensor.type(torch.float32)

def generate_all_onehots(k=4, num_classes=4):
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
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.register_buffer('weight', generate_all_onehots(k,4).float())
        
    def forward(self, input_):
        return F.conv1d(input_, self.weight).eq(self.k).float()

def batch2fasta(batch, file_name):
    """
    Converts a tensor of one-hot sequences into a Fasta file and saves it.

    Parameters
    ----------
    batch : torch tensor
        DESCRIPTION.
    file_name : path and name for fasta file
        DESCRIPTION.

    Returns
    -------
    None.

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

#####################
# PTL Module saving #
#####################

def set_best(my_model, callbacks):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = callbacks['model_checkpoint'].best_model_path
            get_epoch = re.search('epoch=(\d*)', best_path).group(1)
            if 'gs://' in best_path:
                subprocess.call(['gsutil','cp',best_path,tmpdirname])
                best_path = os.path.join( tmpdirname, os.path.basename(best_path) )
            print(f'Best model stashed at: {best_path}', file=sys.stderr)
            print(f'Exists: {os.path.isfile(best_path)}', file=sys.stderr)
            ckpt = torch.load( best_path )
            my_model.load_state_dict( ckpt['state_dict'] )
            print(f'Setting model from epoch: {get_epoch}', file=sys.stderr)
        except KeyError:
            print('Setting most recent model', file=sys.stderr)
    return my_model

def save_model(data_module, model_module, graph_module, 
                model, trainer, args):
    local_dir = args['pl.Trainer'].default_root_dir
    save_dict = {
        'data_module'  : data_module.__name__,
        'data_hparams' : data_module.process_args(args),
        'model_module' : model_module.__name__,
        'model_hparams': model_module.process_args(args),
        'graph_module' : graph_module.__name__,
        'graph_hparams': graph_module.process_args(args),
        'model_state_dict': model.state_dict(),
        'timestamp'    : time.strftime("%Y%m%d_%H%M%S"),
        'random_tag'   : random.randint(100000,999999)
    }
    torch.save(save_dict, os.path.join(local_dir,'torch_checkpoint.pt'))
    
    filename=f'model_artifacts__{save_dict["timestamp"]}__{save_dict["random_tag"]}.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(os.path.join(tmpdirname,filename), 'w:gz') as tar:
            tar.add(local_dir,arcname='artifacts')

        if 'gs://' in args['Main args'].artifact_path:
            clound_target = os.path.join(args['Main args'].artifact_path,filename)
            subprocess.check_call(
                ['gsutil', 'cp', os.path.join(tmpdirname,filename), clound_target]
            )
        else:
            os.makedirs(args['Main args'].artifact_path, exist_ok=True)
            shutil.copy(os.path.join(tmpdirname,filename), args['Main args'].artifact_path)

######################
# PTL Module loading #
######################

def unpack_artifact(artifact_path,download_path='./'):
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

def model_fn(model_dir):
    checkpoint = torch.load(os.path.join(model_dir,'torch_checkpoint.pt'))
    model_module = getattr(_model, checkpoint['model_module'])
    model        = model_module(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    return model

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