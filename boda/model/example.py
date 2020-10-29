import torch
import argparse

class ExampleModel(torch.nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser):
        r"""Argparse interface to expose class kwargs to the command line.
        Args:
            parent_parser (argparse.ArgumentParser): A parent ArgumentParser to 
                which more args will be added.
        Returns:
            An updated ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hiddel_dims', type=int)
        return parser
        
    def __init__(self, data_dir='/no/data_dir/specified', batch_size=8, **kwargs):
        super().__init__()
        delf.data_name = 'ExampleModel'
        self.hiddel_dims  = hiddel_dims
        
    def setup(self, stage):
        r"""A method to setup the model and link torch.nn classes as attributes 
        of `self`. My convention is to think of `criterion` as part of the model.
        Args:
            stage (str): 'fit' or 'test' (e.g., stage of network usage)
        Returns:
            None
        """
        
        self.net = torch.nn.Linear(self.in_features_specified_by_data, self.hiddel_dims)
        sefl.criterion = torch.nn.BCEWithLogitsLoss() # Loss calculations are specified in Graph methods.
        
        raise NotImplementedError
        
    def forward(self, batch):
        r"""A standard `torch` forward call
        Args:
            batch (torch.Tensor): A batch of data (batch_size first).
        Returns:
            A tensor of logits.
        """
        
        hook = self.net(batch)
        return hook
        
        raise NotImplementedError
