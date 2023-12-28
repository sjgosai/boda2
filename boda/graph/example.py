r"""Example of a standardized module to build a computing graph to train DNN 
models. This is based on the LightningModule spec from 
lightning.pytorch.core.module (see https://lightning.ai/docs for more 
information) with additional staticmethods that can add arguments to an 
ArgumentParser which mirror class construction arguments.
"""

import torch
import argparse

def add_optimizer_specific_args(parser, optimizer_name):
    group = parser.add_argument_group('Optimizer args')
    
    if optimizer_name == 'Adadelta':
        group.add_argument('--lr', type=float, default=1.0)
        group.add_argument('--rho',type=float, default=0.9)
        group.add_argument('--eps',type=float, default=1e-6)
        group.add_argument('--weight_decay', type=float, default=0.)
    elif optimizer_name == 'Adam':
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--beta1',type=float, default=0.9)
        group.add_argument('--beta2',type=float, default=0.999)
        group.add_argument('--eps',type=float, default=1e-8)
        group.add_argument('--weight_decay', type=float, default=0.)
        group.add_argument('--amsgrad', type=cutils.str2bool, default=False)
    else:
        group.add_argument('--lr', type=float, default=0.1)
        
    return parser
        
def reorg_optimizer_args(optim_arg_dict):
    """
    Reorganize optimizer-specific argument names.

    Args:
        optim_arg_dict (dict): Dictionary containing optimizer-specific arguments.

    Returns:
        dict: Reorganized optimizer argument dictionary.
    """
    if 'beta1' in optim_arg_dict.keys():
        optim_arg_dict['betas'] = [optim_arg_dict['beta1'], optim_arg_dict['beta2']]
        optim_arg_dict.pop('beta1')
        optim_arg_dict.pop('beta2')
    return optim_arg_dict

class ExampleModel(torch.nn.Module):
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        """
        Add command-line arguments specific to the Graph module.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--optimizer', type=str, default='Adam')
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        """
        Add conditional arguments to the parser based on known arguments.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
            known_args (Namespace): Known arguments.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = add_optimizer_specific_args(parser, known_args.optimizer)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process the command-line arguments for the Graph module.

        Args:
            grouped_args (Namespace): Grouped command-line arguments.

        Returns:
            Namespace: Processed arguments.
        """
        graph_args   = grouped_args['Graph Module args']
        graph_args.optimizer_args = vars(grouped_args['Optimizer args'])
        graph_args.optimizer_args = reorg_optimizer_args(graph_args.optimizer_args)
        return graph_args

    #######################
    # Dead __init__ block #
    #######################
    
    def __init__(self, optimizer='Adam', optimizer_args=None):
        """
        Initialize the ExampleGraph module.

        Args:
            optimizer (str): Name of the optimizer. Default is 'Adam'.
            optimizer_args (dict): Arguments for the optimizer. Default is None.
        """
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        
    #############
    # PTL hooks #
    #############
        
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Union[Optimizer, Tuple[List[Optimizer], List[Dict]]]: Optimizer(s) and scheduler(s).
        """
        self.hpt = hypertune.HyperTune()
        params = [ x for x in self.parameters() if x.requires_grad ]
        print(f'Found {sum(p.numel() for p in params)} parameters')
        optim_class = getattr(torch.optim,self.optimizer)
        my_optimizer= optim_class(self.parameters(), **self.optimizer_args)

        raise NotImplementedError
        return my_optimizer
        
    def training_step(self, batch, batch_idx):
        """
        Training step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the training step.
        """
        x, y   = batch
        y_hat  = self(x)
        loss   = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        raise NotImplementedError
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary containing loss, metric, predictions, and labels for the validation step.
        """
        x, y   = batch
        y_hat = self(x)
        loss   = self.criterion(y_hat, y)
        self.log('valid_loss', loss)
        metric = self.categorical_mse(y_hat, y)

        raise NotImplementedError
        return {'loss': loss, 'metric': metric, 'preds': y_hat, 'labels': y}

    def test_step(self, batch, batch_idx):
        """
        Test step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        raise NotImplementedError
        self.log('test_loss', loss)       
        
