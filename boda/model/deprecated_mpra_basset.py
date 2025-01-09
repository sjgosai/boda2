"""
MIT License

Copyright (c) 2025 Sagar Gosai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import lightning.pytorch as pl

import sys
from .basset import Basset
from ..graph import utils

class MPRA_Basset(pl.LightningModule):
    """
    MPRA_Basset model architecture.

    Args:
        basset_weights_path (str): Path to Basset model weights.
        target_width (int): Width (or length) of the target data.
        learning_rate (float): Learning rate for optimization.
        optimizer (str): Name of the optimizer.
        scheduler (bool): Whether to implement cosine annealing LR scheduler.
        weight_decay (float): Weight decay rate.
        epochs (int): Number of epochs passed to the trainer (used by the scheduler).
        extra_hidden_size (int): Size of the extra hidden layer.
        criterion (str): Loss criterion name.
        last_activation (str): Activation function name for the last layer.
        sneaky_factor (float): Factor for adding the Shannon entropy loss.

    Methods:
        add_model_specific_args(parent_parser): Add model-specific arguments to the argument parser.
        __init__(...): Initialize MPRA_Basset model.
        forward(x): Forward pass through the MPRA_Basset model.
        training_step(batch, batch_idx): Training step implementation.
        validation_step(batch, batch_idx): Validation step implementation.
        test_step(batch, batch_idx): Test step implementation.
        validation_epoch_end(validation_step_outputs): Validation epoch end implementation.
        configure_optimizers(): Configure optimizers and scheduler (if enabled).

    Attributes:
        basset_weights_path (str): Path to Basset model weights.
        target_width (int): Width (or length) of the target data.
        learning_rate (float): Learning rate for optimization.
        optimizer (str): Name of the optimizer.
        scheduler (bool): Whether to implement cosine annealing LR scheduler.
        weight_decay (float): Weight decay rate.
        epochs (int): Number of epochs passed to the trainer (used by the scheduler).
        extra_hidden_size (int): Size of the extra hidden layer.
        sneaky_factor (float): Factor for adding the Shannon entropy loss.
        criterion (nn.Module): Loss criterion module.
        last_activation (nn.Module): Activation function module for the last layer.
        basset_net (Basset): Basset model instance.
        basset_last_hidden_width (int): Width of the last hidden layer in the Basset model.
        output_1, output_2, output_3 (nn.Sequential): Sequential output layers.
        example_input_array (torch.Tensor): Example input array for model visualization.
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the argument parser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added model-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
         
        parser.add_argument('--target_width', type=int, default=3, 
                            help='Width (or length) of the target data') 
        parser.add_argument('--learning_rate', type=float, default=1e-4, 
                            help='Value of the learning rate')
        parser.add_argument('--optimizer', type=str, default='Adam', 
                            help='Name of the optimizer')
        parser.add_argument('--scheduler', type=bool, default=True, 
                            help='If true it implements cosine annealing LR')
        parser.add_argument('--weight_decay', type=float, default=1e-6, 
                            help='Weight decay rate')
        parser.add_argument('--epochs', type=int, default=1, 
                            help='Number of epochs passed to the trainer (used by the scheduler)') 
        return parser
    
    def __init__(self,
                 basset_weights_path=None,
                 target_width=3,
                 learning_rate=1e-4,
                 optimizer='Adam',
                 scheduler=False,
                 weight_decay=1e-6,
                 epochs=1,
                 extra_hidden_size = 100,
                 criterion = 'MSELoss',
                 last_activation='Tanh',
                 sneaky_factor=1,
                 **kwargs):
        """
        Initialize MPRA_Basset model.

        Args:
            basset_weights_path (str): Path to Basset model weights.
            target_width (int): Width (or length) of the target data.
            learning_rate (float): Learning rate for optimization.
            optimizer (str): Name of the optimizer.
            scheduler (bool): Whether to implement cosine annealing LR scheduler.
            weight_decay (float): Weight decay rate.
            epochs (int): Number of epochs passed to the trainer (used by the scheduler).
            extra_hidden_size (int): Size of the extra hidden layer.
            criterion (str): Loss criterion name.
            last_activation (str): Activation function name for the last layer.
            sneaky_factor (float): Factor for adding the Shannon entropy loss.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__()
        self.basset_weights_path = basset_weights_path
        self.target_width = target_width
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.extra_hidden_size = extra_hidden_size
        self.sneaky_factor = sneaky_factor
        
        self.criterion = getattr(nn, criterion)()  
        self.last_activation = getattr(nn, last_activation)()
        
        self.basset_net = Basset()
        if self.basset_weights_path is not None:
            try:
                self.basset_net.load_state_dict(torch.load(self.basset_weights_path))
            except:
                try:
                    self.basset_net.load_state_dict(torch.load(self.basset_weights_path, map_location=torch.device('cpu')))
                except:
                    print('Not able to load Basset weights')
        
        self.basset_last_hidden_width = self.basset_net.linear2_channels

        self.output_1 = nn.Sequential(
            nn.Linear(self.basset_last_hidden_width, self.extra_hidden_size),
            self.last_activation,
            nn.Linear(self.extra_hidden_size, 1)
            )
        
        self.output_2 = nn.Sequential(
            nn.Linear(self.basset_last_hidden_width, self.extra_hidden_size),
            self.last_activation,
            nn.Linear(self.extra_hidden_size, 1)
            )
        
        self.output_3 = nn.Sequential(
            nn.Linear(self.basset_last_hidden_width, self.extra_hidden_size),
            self.last_activation,
            nn.Linear(self.extra_hidden_size, 1)
            )       

        self.example_input_array = torch.rand(1, 4, 600)
        
    def forward(self, x):
        """
        Forward pass through the MPRA_Basset model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model's prediction tensor.
        """
        basset_last_hidden = self.basset_net.decode(self.basset_net.encode(x))
        output_1 = self.output_1(basset_last_hidden)
        output_2 = self.output_2(basset_last_hidden)
        output_3 = self.output_3(basset_last_hidden)
        mpra_pred = torch.cat((output_1, output_2, output_3), dim=1)
        return mpra_pred
        
    def training_step(self, batch, batch_idx):
        """
        Training step implementation.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        y_pred = self(x)
        shannon_pred, shannon_target = utils.shannon_entropy(y_pred), utils.shannon_entropy(y)
        loss = self.criterion(y_pred, y) + self.sneaky_factor*self.criterion(shannon_pred, shannon_target)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation epoch end implementation.

        Args:
            validation_step_outputs: List of validation step outputs.

        Returns:
            None
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'loss': loss, 'pred': y_pred, 'target': y}
        
    def test_step(self, batch, batch_idx):
        """
        Test step implementation.

        Args:
            batch: Test batch.
            batch_idx: Batch index.

        Returns:
            None
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)

    def validation_epoch_end(self, validation_step_outputs):
        """
        Validation epoch end implementation.

        Args:
            validation_step_outputs: List of validation step outputs.

        Returns:
            None
        """
        preds = torch.cat([out['pred'] for out in validation_step_outputs], dim=0)
        targets  = torch.cat([out['target'] for out in validation_step_outputs], dim=0)
        pearsons, mean_pearson = utils.pearson_correlation(preds, targets)
        shannon_pred, shannon_target = utils.shannon_entropy(preds), utils.shannon_entropy(targets)
        specificity_pearson, specificity_mean_pearson = utils.pearson_correlation(shannon_pred, shannon_target)
        self.log('Pearson', mean_pearson)
        self.log('Pearson_Shannon', specificity_mean_pearson)
        res_str = '|'
        res_str += ' Prediction correlation: {:.5f} | Specificity correlation: {:.5f} |' \
                    .format(mean_pearson.item(), specificity_mean_pearson.item())
        print(res_str)
        print('-'*len(res_str))
        
    def configure_optimizers(self):
        """
        Configure optimizers and scheduler (if enabled).

        Returns:
            list: List of optimizers.
        """
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate,
                                                         weight_decay=self.weight_decay)  
        if self.scheduler:
            lr_scheduler = {
                'scheduler' : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6),
                'name': 'learning_rate'
                           }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer