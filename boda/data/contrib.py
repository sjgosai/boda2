### MAKING DATASET CLASS
import torch
import numpy as np
from torch.utils.data import Dataset

class InputSequences(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                sequence, score = line.strip().split('\t')
                self.data.append((sequence, float(score)))

        # Define a mapping for nucleotides to indices
        self.nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence, score = self.data[index]
        # Encode the sequence using one-hot encoding
        sequence_tensor = self.encode_sequence(sequence)

        # Convert score to a torch tensor
        score_tensor = torch.tensor(score, dtype=torch.float32)

        return sequence_tensor, score_tensor

    def encode_sequence(self, sequence):
        # Initialize an array of zeros with shape (4, sequence_length),
        # where sequence_length is the length of the DNA sequence (in this case, 200)
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.core import LightningDataModule

class SeqDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Data Module args')

        group.add_argument('--train_file', type=str, required=True)
        group.add_argument('--val_file', type=str, required=True)
        group.add_argument('--test_file', type=str, required=True)
        group.add_argument('--batch_size', type=int, required=True)
        group.add_argument('--left_flank', type=str, default=boda.common.constants.MPRA_UPSTREAM[-200:])
        group.add_argument('--right_flank', type=str, default=boda.common.constants.MPRA_DOWNSTREAM[:200])
        return parser

    def __init__(self, train_file, val_file, test_file, batch_size=10, left_flank='', right_flank=''):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load the datasets from the files
        self.train_dataset = InputSequences(self.train_file)
        self.val_dataset = InputSequences(self.val_file)
        self.test_dataset = InputSequences(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)