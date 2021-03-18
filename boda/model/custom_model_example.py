import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pytorch_lightning as pl

'''
Custom MPRA activity predictor
'''
class MPRAregressionModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        #training params
        parser.add_argument('--LR', type=float, default=0.0005)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weightDecay', type=float, default=1e-8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--scheduler', type=bool, default=True)
        
        #input shape
        parser.add_argument('--seqLen', type=int, default=600)
        parser.add_argument('--numFeatures', type=int, default=4)
        parser.add_argument('--targetLen', type=int, default=1)
        
        #network params
        parser.add_argument('--numChannles1', type=int, default=20)
        parser.add_argument('--kernelSize1', type=int, default=6)
        parser.add_argument('--stride1', type=int, default=3)
        parser.add_argument('--padding1', type=int, default=0)
        parser.add_argument('--dilation1', type=int, default=1)
        
        parser.add_argument('--poolKernel1', type=int, default=4)
        parser.add_argument('--poolStride1', type=int, default=2)
        
        parser.add_argument('--numChannles2', type=int, default=10)
        parser.add_argument('--kernelSize2', type=int, default=4)
        parser.add_argument('--stride2', type=int, default=2)
        parser.add_argument('--padding2', type=int, default=0)
        parser.add_argument('--dilation2', type=int, default=1)
        
        parser.add_argument('--poolKernel2', type=int, default=2)
        parser.add_argument('--poolStride2', type=int, default=2)
        
        parser.add_argument('--linearLayerLen1', type=int, default=50)
        parser.add_argument('--linearLayerLen2', type=int, default=10)
        
        args = parser.parse_args()
        print(f'Parser argumentss: {vars(args)}')
        return parser
    
    def __init__(self,
                 LR=0.0005,
                 momentum=0.9,
                 weightDecay=1e-8,
                 dropout=0.2,
                 optimizer='Adam',
                 scheduler=True,
                 seqLen=600,
                 numFeatures=4,
                 targetLen=1,
                 numChannles1=20,
                 kernelSize1=6,
                 stride1=3,
                 padding1=0,
                 dilation1=1,
                 poolKernel1=4,
                 poolStride1=2,
                 numChannles2=10,
                 kernelSize2=4,
                 stride2=2,
                 padding2=0,
                 dilation2=1,
                 poolKernel2=2,
                 poolStride2=2,
                 linearLayerLen1=50,
                 linearLayerLen2=10, **kwargs):
        
        super().__init__()
        #training params
        self.LR = LR
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.dropout = dropout       
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = nn.MSELoss()
        
        #input shape
        self.seqLen = seqLen
        self.numFeatures = numFeatures
        self.targetLen = targetLen
        self.example_input_array = torch.rand(1, self.numFeatures, self.seqLen)
        
        #network params
        self.numChannels1 = numChannles1
        self.kernelSize1 = kernelSize1
        self.stride1 = stride1
        self.padding1 = padding1
        self.dilation1 = dilation1
        self.convLayerLen1 = self.conv1D_OutDim(L_in=self.seqLen, 
                                       kernel_size=self.kernelSize1,
                                       stride=self.stride1,
                                       padding=self.padding1,
                                       dilation=self.dilation1)
        
        self.poolKernel1 = poolKernel1
        self.poolStride1 = poolStride1
        self.poolLayerLen1 = self.conv1D_OutDim(L_in=self.convLayerLen1, 
                                       kernel_size=self.poolKernel1,
                                       stride=self.poolStride1)
        
        self.numChannels2 = numChannles2
        self.kernelSize2 = kernelSize2
        self.stride2 = stride2
        self.padding2 = padding2
        self.dilation2 = dilation2
        self.convLayerLen2 = self.conv1D_OutDim(L_in=self.poolLayerLen1, 
                                       kernel_size=self.kernelSize2,
                                       stride=self.stride2,
                                       padding=self.padding2,
                                       dilation=self.dilation2)    
        
        self.poolKernel2 = poolKernel2
        self.poolStride2 = poolStride2
        self.poolLayerLen2 = self.conv1D_OutDim(L_in=self.convLayerLen2, 
                                       kernel_size=self.poolKernel2,
                                       stride=self.poolStride2)
        
        self.linearLayerLen1 = linearLayerLen1
        self.linearLayerLen2 = linearLayerLen2
        
        #hidden layers
        self.convLayers = nn.Sequential(
                nn.Conv1d(in_channels=self.numFeatures, out_channels=self.numChannels1,
                          kernel_size=self.kernelSize1, stride=self.stride1,
                          padding=self.padding1, dilation=self.dilation1),
                torch.nn.Tanh(),
                torch.nn.MaxPool1d(kernel_size=self.poolKernel1, stride=self.poolStride1),
                nn.Conv1d(in_channels=self.numChannels1, out_channels=self.numChannels2,
                          kernel_size=self.kernelSize2, stride=self.stride2,
                          padding=self.padding2, dilation=self.dilation2),
                torch.nn.Tanh(),
                torch.nn.MaxPool1d(kernel_size=self.poolKernel2, stride=self.poolStride2) )
        
        self.linearLayers = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.poolLayerLen2 * self.numChannels2, self.linearLayerLen1),
                nn.Tanh(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.linearLayerLen1, self.linearLayerLen2),
                nn.Tanh(),
                nn.Dropout(p=0.5*self.dropout),
                nn.Linear(self.linearLayerLen2, self.targetLen) )
 
    def forward(self, x):
        convOutput = self.convLayers(x)
        linearInput = torch.flatten(convOutput, start_dim=1) 
        y_hat = self.linearLayers(linearInput)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.LR, betas=(0.9, 0.999), 
                                 eps=1e-8, weight_decay = self.weightDecay)
        elif self.optimizer == 'RMS':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.LR, alpha=0.99, eps=1e-8,
                                        weight_decay = self.weightDecay, momentum = self.momentum,
                                        centered=True)
        
        #needs a better way of passing the number of epochs
        if self.scheduler:
            lr_scheduler = {
                'scheduler' : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs,   
                                                           eta_min=0.0000001, last_epoch=-1),
                'name': 'learning_rate'
                           }
            return [optimizer], [lr_scheduler]
        
        else:
            return optimizer
    
    
    #calculate the output length of a 1D-convolutional layer    
    @staticmethod
    def conv1D_OutDim(L_in, kernel_size, stride, padding=0, dilation=1):
        return (L_in + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1     
    
  
    
#------------------------------- EXAMPLE --------------------------------------------------
if __name__ == '__main__':   
    
    #OPTION 1: use add_model_specific_args to pass arguments:
    parser = argparse.ArgumentParser(description="MPRAmodel", add_help=False)
    parser = MPRAregressionModel.add_model_specific_args(parser) 
    model = MPRAregressionModel(parser)
    summary(model, (model.numFeatures, model.seqLen) )
    
    #OPTION 2: pass the arguments manually:
    # model = MPRAregressionModel(LR=0.0005,
    #                             momentum=0.9,
    #                             weightDecay=1e-8,
    #                             dropout=0.2,
    #                             optimizer='Adam',
    #                             scheduler=True,
    #                             seqLen=600,
    #                             numFeatures=4,
    #                             targetLen=1,
    #                             numChannles1=20,
    #                             kernelSize1=6,
    #                             stride1=3,
    #                             padding1=0,
    #                             dilation1=1,
    #                             poolKernel1=4,
    #                             poolStride1=2,
    #                             numChannles2=10,
    #                             kernelSize2=4,
    #                             stride2=2,
    #                             padding2=0,
    #                             dilation2=1,
    #                             poolKernel2=2,
    #                             poolStride2=2,
    #                             linearLayerLen1=50,
    #                             linearLayerLen2=10)
    # summary(model, (model.numFeatures, model.seqLen) )
