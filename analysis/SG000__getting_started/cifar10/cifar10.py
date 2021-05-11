import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as ptl

# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class MyNet(ptl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--conv1_width', type=int, default=6)
        parser.add_argument('--conv2_width', type=int, default=16)
        parser.add_argument('--fc1_width', type=int, default=120)
        parser.add_argument('--fc2_width', type=int, default=84)
        parser.add_argument('--dropout1', type=float, default=0.5)
        parser.add_argument('--dropout2', type=float, default=0.5)
        
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        
        return parser
    
    def __init__(self, conv1_width=6, conv2_width=16, 
                 fc1_width=120, fc2_width=84, 
                 dropout1=0.5, dropout2=0.5, 
                 learning_rate=1e-3, **kwargs):
        super().__init__()
                
        self.conv1_width = conv1_width
        self.conv2_width = conv2_width
        self.fc1_width = fc1_width
        self.fc2_width = fc2_width
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        
        self.unused_kwargs = kwargs
        self.save_hyperparameters()
        
        self.is_built = False
        
    def setup(self, step):
        if not self.is_built:
            print('Setting up a super-fresh new model.')
            self.conv1 = nn.Conv2d(3, self.conv1_width, 5)
            self.pool  = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(self.conv1_width, self.conv2_width, 5)
            self.fc1   = nn.Linear(self.conv2_width * 5 * 5, self.fc1_width)
            self.drop1 = nn.Dropout(p=self.dropout1)
            self.fc2   = nn.Linear(self.fc1_width, self.fc2_width)
            self.drop2 = nn.Dropout(p=self.dropout2)
            self.fc3   = nn.Linear(self.fc2_width, 10)

            self.criterion = nn.CrossEntropyLoss()
            
            self.is_built = True
        else:
            print('A model has already been setup.')
            pass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
class MyDataModule(ptl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--data_dir', type=str, default='/tmp/')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--workers', type=int, default=1)
                
        return parser

    def __init__(self, data_dir='/tmp/', batch_size=16, workers=1, **kwargs):
        super().__init__()
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.workers    = workers
        
        self.unused_kwargs = kwargs
        #self.save_hyperparameters()
        
    def prepare_data(self):
        cifar10_data = torchvision.datasets.CIFAR10(root=self.data_dir, download=True)
        return None
    
    def setup(self, step):
        transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
        self.train = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                  download=False, transform=transform)
        self.valid = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                  download=False, transform=transform)
        return None
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, 
                          shuffle=True,  num_workers=self.workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers)
