r"""Example of a standardized module to build DNN models. This is based on 
the standard PyTorch spec with additional staticmethods that can add 
arguments to an ArgumentParser which mirror class construction arguments.
"""

import torch
import argparse

class ExampleModel(torch.nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser):
        r"""Argparse interface to expose class constructor args to the 
        command line.
        Args:
            parent_parser (argparse.ArgumentParser): A parent ArgumentParser to 
                which more args will be added.
        Returns:
            An updated ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Model Module args')
        group.add_argument('--hiddel_dims', type=int, default=8)
        group.add_argument('--output_dims', type=int, default=1)
        group.add_argument('--activation', type=str, default='ReLU')
        
        return parser
        
    @staticmethod
    def add_conditional_args(parser, known_args):
        r"""Method to add conditional args after `add_model_specific args` 
        sets primary arguments. Required to specify for CLI conpatability, 
        but can return unmodified parser as in this example. These args should 
        mirror class construction arguments that would be determined based on 
        other arguments.
        Args:
            parser (argparse.AtgumentParser): an ArgumentParser to  which 
                more args can be added.
            known_args (Namespace): Known arguments that have been parsed 
                by non-conditionally specified args.
        """
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Perform any required processessing of command line args required 
        before passing to the class constructor.

        Args:
            grouped_args (Namespace): Namespace of known arguments with 
            `'Model Module args'` key.

        Returns:
            Namespace: A modified namespace that can be passed to the 
            associated class constructor.
        """
        model_args   = grouped_args['Model Module args']
        return model_args

    def __init__(self, hidden_dims=8, output_dims=1, activation='ReLU', **kwargs):
        super().__init__()
        self.model_name = 'ExampleModel'
        self.hidden_dims  = hidden_dims
        self.output_dims  = output_dims
        self.activation   = activation
        self.hidden_layer = torch.nn.Linear(self.in_features_specified_by_data, self.hidden_dims)
        self.output_layer = torch.nn.Linear(self.hidden_dims, self.output_dims)
        self.hidden_activation = getattr(torch.nn, self.activation)
        self.criterion = torch.nn.BCEWithLogitsLoss()
                
    def forward(self, batch):
        r"""A standard `torch` forward call
        Args:
            batch (torch.Tensor): A batch of data (batch_size first).
        Returns:
            A tensor of logits.
        """
        
        hook = self.hidden_layer(batch)
        hook = self.hidden_activation( hook )
        hook = self.output_layer( hook )
        return hook
        
        raise NotImplementedError
