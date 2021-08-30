import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import pytorch_lightning as pl

import sys
from .basset import Basset

class MPRA_Basset(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
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
        Parameters
        ----------
        basset_weights_path : TYPE, optional
            DESCRIPTION. The default is None.
        target_width : TYPE, optional
            DESCRIPTION. The default is 3.
        learning_rate : TYPE, optional
            DESCRIPTION. The default is 1e-4.
        optimizer : TYPE, optional
            DESCRIPTION. The default is 'Adam'.
        scheduler : TYPE, optional
            DESCRIPTION. The default is False.
        weight_decay : TYPE, optional
            DESCRIPTION. The default is 1e-6.
        epochs : TYPE, optional
            DESCRIPTION. The default is 1.
        extra_hidden_size : TYPE, optional
            DESCRIPTION. The default is 100.
        criterion : TYPE, optional
            DESCRIPTION. The default is 'MSELoss'.
        last_activation : TYPE, optional
            DESCRIPTION. The default is 'Tanh'.
        sneaky_factor : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

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
        basset_last_hidden = self.basset_net.decode(self.basset_net.encode(x))
        output_1 = self.output_1(basset_last_hidden)
        output_2 = self.output_2(basset_last_hidden)
        output_3 = self.output_3(basset_last_hidden)
        mpra_pred = torch.cat((output_1, output_2, output_3), dim=1)
        return mpra_pred
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        shannon_pred, shannon_target = Shannon_entropy(y_pred), Shannon_entropy(y)
        loss = self.criterion(y_pred, y) + self.sneaky_factor*self.criterion(shannon_pred, shannon_target)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'loss': loss, 'pred': y_pred, 'target': y}
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)

    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat([out['pred'] for out in validation_step_outputs], dim=0)
        targets  = torch.cat([out['target'] for out in validation_step_outputs], dim=0)
        pearsons, mean_pearson = Pearson_correlation(preds, targets)
        shannon_pred, shannon_target = Shannon_entropy(preds), Shannon_entropy(targets)
        specificity_pearson, specificity_mean_pearson = Pearson_correlation(shannon_pred, shannon_target)
        self.log('Pearson', mean_pearson)
        self.log('Pearson_Shannon', specificity_mean_pearson)
        res_str = '|'
        res_str += ' Prediction correlation: {:.5f} | Specificity correlation: {:.5f} |' \
                    .format(mean_pearson.item(), specificity_mean_pearson.item())
        print(res_str)
        print('-'*len(res_str))
        
    def configure_optimizers(self):
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