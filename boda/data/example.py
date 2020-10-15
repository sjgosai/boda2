import torch
import argparse

class ExampleData(torch.nn.Module):
    @staticmethod
    def add_data_specific_args(parent_parser):
        r"""Argparse interface to expose class kwargs to the command line.
        Args:
            parent_parser (argparse.ArgumentParser): A parent ArgumentParser to 
                which more args will be added.
        Returns:
            An updated ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--batch_size', type=int)
        return parser
        
    def __init__(self, data_dir='/no/data_dir/specified', batch_size=8, **kwargs):
        super().__init__()
        delf.data_name = 'ExampleData'
        self.data_dir  = data_dir
        self.batch_size= batch_size
        
        self.unused_kwargs = kwargs
        
    def prepare_data(self):
        r"""A method to prepare data based on class attributes, usually by 
        linking new attributes to `self`.
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
        
    def train_dataloader(self):
        r"""Method that returns a DataLoader for training data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
        return Training_DataLoader
        
    def val_dataloader(self):
        r"""Method that returns a DataLoader for validation data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
        return Validation_DataLoader
        
    def test_dataloader(self):
        r"""Method that returns a DataLoader for testing data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
        return Testing_DataLoader
        
