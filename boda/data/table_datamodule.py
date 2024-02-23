import torch
import numpy as np
from torch.utils.data import Dataset

from ..common import constants, utils


class InputSequences(Dataset):
    def __init__(self, file_path, left_flank, right_flank, seq_len=600, use_revcomp=False, skip_header=False):
        self.data = []
        self.left_flank = left_flank
        self.right_flank = right_flank
        self.seq_len = seq_len
        self.use_revcomp = use_revcomp
        self.skip_header = skip_header
        
        with open(file_path, 'r') as file:
            if self.skip_header:
                file.readline()
            for line in file:
                parts = line.strip().split('\t')
                sequence = parts[0]
                score = list(map(float, parts[1:]))  # Convert all columns after the first one to floats
                self.data.append((sequence, score))

        # Define a mapping for nucleotides to indices
        self.nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        if self.use_revcomp:
            return 2*len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.use_revcomp:
            use_index = index // 2
        else:
            use_index = index
        sequence, score = self.data[use_index]
        
        # Add left and right flanks to the sequence
        left_len = (self.seq_len - len(sequence)) // 2
        right_len= self.seq_len - (len(sequence) + left_len)
        sequence_with_flanks = self.left_flank[-left_len:] + sequence + self.right_flank[:right_len]
        
        # Encode the sequence using one-hot encoding
        sequence_tensor = self.encode_sequence(sequence_with_flanks)

        # Convert score to a torch tensor
        score_tensor = torch.tensor(score, dtype=torch.float32)

        if self.use_revcomp and index % 2 == 1:
            sequence_tensor = sequence_tensor.flip(dims=[0,1])
        
        return sequence_tensor, score_tensor

    def encode_sequence(self, sequence):
        # Initialize an array of zeros with shape (4, sequence_length),
        # where sequence_length is the length of the DNA sequence (in this case, len(sequence))
        one_hot = np.zeros((4, len(sequence)))

        # Convert each nucleotide in the sequence to its one-hot representation
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_to_index:
                index = self.nucleotide_to_index[nucleotide]
                one_hot[index, i] = 1

        # Convert the numpy array to a torch tensor
        sequence_tensor = torch.tensor(one_hot, dtype=torch.float32)

        return sequence_tensor

### DATAMODULE
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch import LightningDataModule

class SeqDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Data Module args')

        group.add_argument('--train_file', type=str, required=True)
        group.add_argument('--val_file', type=str, required=True)
        group.add_argument('--test_file', type=str, required=True)
        group.add_argument('--batch_size', type=int, required=True)
        group.add_argument('--left_flank', type=str, default=constants.MPRA_UPSTREAM)
        group.add_argument('--right_flank', type=str, default=constants.MPRA_DOWNSTREAM)
        group.add_argument('--seq_len', type=int, default=600)
        group.add_argument('--use_revcomp', type=utils.str2bool, default=False)
        group.add_argument('--skip_header', type=utils.str2bool, default=False)
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser
    
    @staticmethod
    def process_args(grouped_args):
        data_args    = grouped_args['Data Module args']
        return data_args

    def __init__(self, train_file, val_file, test_file, batch_size=10, left_flank='', right_flank='', seq_len=600, use_revcomp=False, skip_header=False):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.left_flank = left_flank
        self.right_flank = right_flank
        self.seq_len = seq_len
        self.use_revcomp = use_revcomp
        self.skip_header = skip_header
        
    def setup(self, stage=None):
        # Load the datasets from the files
        self.train_dataset = InputSequences(self.train_file, self.left_flank, self.right_flank, self.seq_len, self.use_revcomp, self.skip_header)
        self.val_dataset = InputSequences(self.val_file, self.left_flank, self.right_flank, self.seq_len, self.use_revcomp, self.skip_header)
        self.test_dataset = InputSequences(self.test_file, self.left_flank, self.right_flank, self.seq_len, self.use_revcomp, self.skip_header)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
