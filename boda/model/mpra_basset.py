import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import pytorch_lightning as pl
#from basset import Basset

import sys
sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
import boda
from boda.model.basset import Basset

class MPRA_Basset(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        args = parser.parse_args()
        print(f'Parser argumentss: {vars(args)}')
        return parser
    
    def __init__(self,
                 pretrained=True,
                 basset_weights_path=None,
                 target_width=3,
                 learning_rate=1e-4,
                 optimizer='Adam',
                 scheduler=False,
                 weight_decay=1e-6,
                 epochs=1,
                 **kwargs):
        
        super().__init__()
        self.pretrained = pretrained
        self.basset_weights_path = basset_weights_path
        self.target_width = target_width
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        self.criterion = nn.MSELoss()       
        self.basset_net = Basset()
        if self.pretrained:
            try:
                self.basset_net.load_state_dict(torch.load(self.basset_weights_path))
            except:
                self.basset_net.load_state_dict(torch.load(self.basset_weights_path, map_location=torch.device('cpu')))
        
        self.basset_last_hidden_width = self.basset_net.linear2_channels
        self.mpra_output = nn.Linear(self.basset_last_hidden_width, self.target_width)
        
    def forward(self, x):
        _, basset_last_hidden = self.basset_net(x)
        mpra_pred = self.mpra_output(basset_last_hidden)
        return mpra_pred
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)       
        
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
        
        
#------------------------------- EXAMPLE --------------------------------------------------
if __name__ == '__main__':   
    pl.seed_everything(1)
    model = MPRA_Basset(basset_weights_path='./my-model.epoch_5-step_19885.pkl')
    summary(model, (4, 600))