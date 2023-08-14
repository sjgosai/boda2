import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as ptl

import sys
from ..common import utils 

def get_padding(kernel_size):
    """
    Calculate padding values for convolutional layers.

    Args:
        kernel_size (int): Size of the convolutional kernel.

    Returns:
        list: Padding values for left and right sides of the kernel.
    """
    left = (kernel_size - 1) // 2
    right= kernel_size - 1 - left
    return [ max(0,x) for x in [left,right] ]

class Conv1dNorm(nn.Module):
    """
    Convolutional layer with optional normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation rate of the convolution.
        groups (int): Number of groups for grouped convolution.
        bias (bool): Whether to include bias terms.
        batch_norm (bool): Whether to use batch normalization.
        weight_norm (bool): Whether to use weight normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, batch_norm=True, weight_norm=True):
        """
        Initialize Conv1dNorm layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding for the convolution.
            dilation (int): Dilation rate of the convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to include bias terms.
            batch_norm (bool): Whether to use batch normalization.
            weight_norm (bool): Whether to use weight normalization.
        """
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
    def forward(self, input):
        """
        Forward pass through the convolutional layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        try:
            return self.bn_layer( self.conv( input ) )
        except AttributeError:
            return self.conv( input )
        
class LinearNorm(nn.Module):
    """
    Linear layer with optional normalization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include bias terms.
        batch_norm (bool): Whether to use batch normalization.
        weight_norm (bool): Whether to use weight normalization.
    """
    def __init__(self, in_features, out_features, bias=True, 
                 batch_norm=True, weight_norm=True):
        """
        Initialize LinearNorm layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include bias terms.
            batch_norm (bool): Whether to use batch normalization.
            weight_norm (bool): Whether to use weight normalization.
        """
        super(LinearNorm, self).__init__()
        self.linear  = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
    def forward(self, input):
        """
        Forward pass through the linear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        try:
            return self.bn_layer( self.linear( input ) )
        except AttributeError:
            return self.linear( input )

class Basset(ptl.LightningModule):
    """
    Basset model architecture.

    Args:
        conv1_channels (int): Number of output channels in the first convolutional layer.
        conv1_kernel_size (int): Kernel size of the first convolutional layer.
        conv2_channels (int): Number of output channels in the second convolutional layer.
        conv2_kernel_size (int): Kernel size of the second convolutional layer.
        conv3_channels (int): Number of output channels in the third convolutional layer.
        conv3_kernel_size (int): Kernel size of the third convolutional layer.
        linear1_channels (int): Number of output channels in the first linear layer.
        linear2_channels (int): Number of output channels in the second linear layer.
        n_outputs (int): Number of output classes.
        activation (str): Activation function name.
        dropout_p (float): Dropout probability.
        use_batch_norm (bool): Whether to use batch normalization.
        use_weight_norm (bool): Whether to use weight normalization.
        loss_criterion (str): Loss criterion name.

    Methods:
        add_model_specific_args(parent_parser): Add model-specific arguments to the argument parser.
        add_conditional_args(parser, known_args): Add conditional arguments based on known arguments.
        process_args(grouped_args): Process grouped arguments and return model-specific arguments.
        encode(x): Encode input through the Basset model's encoding layers.
        decode(x): Decode encoded tensor through the Basset model's decoding layers.
        classify(x): Classify decoded tensor using the Basset model's classification layer.
        forward(x): Forward pass through the Basset model.
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
        group  = parser.add_argument_group('Model Module args')
        
        group.add_argument('--conv1_channels', type=int, default=300)
        group.add_argument('--conv1_kernel_size', type=int, default=19)
        
        group.add_argument('--conv2_channels', type=int, default=200)
        group.add_argument('--conv2_kernel_size', type=int, default=11)
        
        group.add_argument('--conv3_channels', type=int, default=200)
        group.add_argument('--conv3_kernel_size', type=int, default=7)
        
        group.add_argument('--linear1_channels', type=int, default=1000)
        group.add_argument('--linear2_channels', type=int, default=1000)
        group.add_argument('--n_outputs', type=int, default=280)
        
        group.add_argument('--dropout_p', type=float, default=0.3)
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True)
        group.add_argument('--use_weight_norm',type=utils.str2bool, default=False)
        
        group.add_argument('--loss_criterion',type=str, default='CrossEntropyLoss')
        
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Add conditional arguments based on known arguments.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
            known_args (Namespace): Namespace of known arguments.

        Returns:
            argparse.ArgumentParser: Argument parser with added conditional arguments.
        """
        model_args   = grouped_args['Model Module args']
        return model_args

    def __init__(self, conv1_channels=300, conv1_kernel_size=19, 
                 conv2_channels=200, conv2_kernel_size=11, 
                 conv3_channels=200, conv3_kernel_size=7, 
                 linear1_channels=1000, linear2_channels=1000, 
                 n_outputs=280, activation='ReLU', 
                 dropout_p=0.3, use_batch_norm=True, use_weight_norm=False,
                 loss_criterion='CrossEntropyLoss'):
        """
        Initialize Basset model.

        Args:
            conv1_channels (int): Number of output channels in the first convolutional layer.
            conv1_kernel_size (int): Kernel size of the first convolutional layer.
            conv2_channels (int): Number of output channels in the second convolutional layer.
            conv2_kernel_size (int): Kernel size of the second convolutional layer.
            conv3_channels (int): Number of output channels in the third convolutional layer.
            conv3_kernel_size (int): Kernel size of the third convolutional layer.
            linear1_channels (int): Number of output channels in the first linear layer.
            linear2_channels (int): Number of output channels in the second linear layer.
            n_outputs (int): Number of output classes.
            activation (str): Activation function name.
            dropout_p (float): Dropout probability.
            use_batch_norm (bool): Whether to use batch normalization.
            use_weight_norm (bool): Whether to use weight normalization.
            loss_criterion (str): Loss criterion name.
        """                                         
        super().__init__()        
        
        self.conv1_channels    = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        
        self.conv2_channels    = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        
        self.conv3_channels    = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        
        self.linear1_channels  = linear1_channels
        self.linear2_channels  = linear2_channels
        self.n_outputs         = n_outputs
        
        self.activation        = activation
        
        self.dropout_p         = dropout_p
        self.use_batch_norm    = use_batch_norm
        self.use_weight_norm   = use_weight_norm
        
        self.loss_criterion    = loss_criterion
        
        self.pad1  = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(4, 
                                self.conv1_channels, self.conv1_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad2  = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, 
                                self.conv2_channels, self.conv2_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad3  = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, 
                                self.conv3_channels, self.conv3_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        
        self.pad4 = nn.ConstantPad1d((1,1), 0.)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        
        self.linear1 = LinearNorm(self.conv3_channels*13, self.linear1_channels, 
                                  bias=True, 
                                  batch_norm=self.use_batch_norm, 
                                  weight_norm=self.use_weight_norm)
        self.linear2 = LinearNorm(self.linear1_channels, self.linear2_channels, 
                                  bias=True, 
                                  batch_norm=self.use_batch_norm, 
                                  weight_norm=self.use_weight_norm)
        self.output  = nn.Linear(self.linear2_channels, self.n_outputs)
        
        self.nonlin  = getattr(nn, self.activation)()                               
        
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.criterion = getattr(nn,self.loss_criterion)()
        
    def encode(self, x):
        """
        Encode input through the Basset model's encoding layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        hook = self.nonlin( self.conv1( self.pad1( x ) ) )
        hook = self.maxpool_3( hook )
        hook = self.nonlin( self.conv2( self.pad2( hook ) ) )
        hook = self.maxpool_4( hook )
        hook = self.nonlin( self.conv3( self.pad3( hook ) ) )
        hook = self.maxpool_4( self.pad4( hook ) )        
        hook = torch.flatten( hook, start_dim=1 )
        return hook
    
    def decode(self, x):
        """
        Decode encoded tensor through the Basset model's decoding layers.

        Args:
            x (torch.Tensor): Encoded tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        hook = self.dropout( self.nonlin( self.linear1( x ) ) )
        hook = self.dropout( self.nonlin( self.linear2( hook ) ) )
        return hook
    
    def classify(self, x):
        """
        Classify decoded tensor using the Basset model's classification layer.

        Args:
            x (torch.Tensor): Decoded tensor.

        Returns:
            torch.Tensor: Classification output tensor.
        """
        output = self.output( x )
        return output
        
    def forward(self, x):
        """
        Forward pass through the Basset model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output

class BassetVL(ptl.LightningModule):
    """
    BassetVL (Variant of Basset with Variable Linear Layers) model architecture.

    Args:
        conv1_channels (int): Number of output channels in the first convolutional layer.
        conv1_kernel_size (int): Kernel size of the first convolutional layer.
        conv2_channels (int): Number of output channels in the second convolutional layer.
        conv2_kernel_size (int): Kernel size of the second convolutional layer.
        conv3_channels (int): Number of output channels in the third convolutional layer.
        conv3_kernel_size (int): Kernel size of the third convolutional layer.
        n_linear_layers (int): Number of linear layers.
        linear_channels (int): Number of output channels in linear layers.
        n_outputs (int): Number of output classes.
        activation (str): Activation function name.
        dropout_p (float): Dropout probability.
        use_batch_norm (bool): Whether to use batch normalization.
        use_weight_norm (bool): Whether to use weight normalization.
        loss_criterion (str): Loss criterion name.

    Methods:
        add_model_specific_args(parent_parser): Add model-specific arguments to the argument parser.
        add_conditional_args(parser, known_args): Add conditional arguments based on known arguments.
        process_args(grouped_args): Process grouped arguments to extract model-specific arguments.
        encode(x): Encode input through the BassetVL model's encoding layers.
        decode(x): Decode encoded tensor through the BassetVL model's decoding layers.
        classify(x): Classify decoded tensor using the BassetVL model's classification layer.
        forward(x): Forward pass through the BassetVL model.
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
        group  = parser.add_argument_group('Model Module args')
        
        group.add_argument('--conv1_channels', type=int, default=300)
        group.add_argument('--conv1_kernel_size', type=int, default=19)
        
        group.add_argument('--conv2_channels', type=int, default=200)
        group.add_argument('--conv2_kernel_size', type=int, default=11)
        
        group.add_argument('--conv3_channels', type=int, default=200)
        group.add_argument('--conv3_kernel_size', type=int, default=7)
        
        group.add_argument('--n_linear_layers', type=int, default=2)
        group.add_argument('--linear_channels', type=int, default=1000)
        group.add_argument('--n_outputs', type=int, default=280)
        
        group.add_argument('--dropout_p', type=float, default=0.3)
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True)
        group.add_argument('--use_weight_norm',type=utils.str2bool, default=False)
        
        group.add_argument('--loss_criterion',type=str, default='CrossEntropyLoss')
        
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        """
        Add conditional arguments based on known arguments.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
            known_args (Namespace): Namespace of known arguments.

        Returns:
            argparse.ArgumentParser: Argument parser with added conditional arguments.
        """
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process grouped arguments to extract model-specific arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            dict: Model-specific arguments.
        """
        model_args   = grouped_args['Model Module args']
        return model_args

    def __init__(self, conv1_channels=300, conv1_kernel_size=19, 
                 conv2_channels=200, conv2_kernel_size=11, 
                 conv3_channels=200, conv3_kernel_size=7, 
                 n_linear_layers=2, linear_channels=1000, 
                 n_outputs=280, activation='ReLU', 
                 dropout_p=0.3, use_batch_norm=True, use_weight_norm=False,
                 loss_criterion='MSELoss'):   
        """
        Initialize BassetVL model.

        Args:
            conv1_channels (int): Number of output channels in the first convolutional layer.
            conv1_kernel_size (int): Kernel size of the first convolutional layer.
            conv2_channels (int): Number of output channels in the second convolutional layer.
            conv2_kernel_size (int): Kernel size of the second convolutional layer.
            conv3_channels (int): Number of output channels in the third convolutional layer.
            conv3_kernel_size (int): Kernel size of the third convolutional layer.
            n_linear_layers (int): Number of linear layers.
            linear_channels (int): Number of output channels in linear layers.
            n_outputs (int): Number of output classes.
            activation (str): Activation function name.
            dropout_p (float): Dropout probability.
            use_batch_norm (bool): Whether to use batch normalization.
            use_weight_norm (bool): Whether to use weight normalization.
            loss_criterion (str): Loss criterion name.
        """                                             
        super().__init__()        
        
        self.conv1_channels    = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        
        self.conv2_channels    = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        
        self.conv3_channels    = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        
        self.n_linear_layers   = n_linear_layers
        self.linear_channels   = linear_channels
        self.n_outputs         = n_outputs
        
        self.activation        = activation
        
        self.dropout_p         = dropout_p
        self.use_batch_norm    = use_batch_norm
        self.use_weight_norm   = use_weight_norm
        
        self.loss_criterion    = loss_criterion
        
        self.pad1  = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(4, 
                                self.conv1_channels, self.conv1_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad2  = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, 
                                self.conv2_channels, self.conv2_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad3  = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, 
                                self.conv3_channels, self.conv3_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        
        self.pad4 = nn.ConstantPad1d((1,1), 0.)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        
        next_in_channels = self.conv3_channels*13
        
        for i in range(self.n_linear_layers):
            
            setattr(self, f'linear{i+1}', 
                    LinearNorm(next_in_channels, self.linear_channels, 
                               bias=True, 
                               batch_norm=self.use_batch_norm, 
                               weight_norm=self.use_weight_norm)
                   )
            next_in_channels = self.linear_channels

        self.output  = nn.Linear(next_in_channels, self.n_outputs)
        
        self.nonlin  = getattr(nn, self.activation)()                               
        
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.criterion = getattr(nn,self.loss_criterion)()
        
    def encode(self, x):
        """
        Encode input through the BassetVL model's encoding layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        hook = self.nonlin( self.conv1( self.pad1( x ) ) )
        hook = self.maxpool_3( hook )
        hook = self.nonlin( self.conv2( self.pad2( hook ) ) )
        hook = self.maxpool_4( hook )
        hook = self.nonlin( self.conv3( self.pad3( hook ) ) )
        hook = self.maxpool_4( self.pad4( hook ) )        
        hook = torch.flatten( hook, start_dim=1 )
        return hook
    
    def decode(self, x):
        """
        Decode encoded tensor through the BassetVL model's decoding layers.

        Args:
            x (torch.Tensor): Encoded tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout( 
                self.nonlin( 
                    getattr(self,f'linear{i+1}')(hook)
                )
            )
        return hook
    
    def classify(self, x):
        """
        Classify decoded tensor using the BassetVL model's classification layer.

        Args:
            x (torch.Tensor): Decoded tensor.

        Returns:
            torch.Tensor: Classification output tensor.
        """
        output = self.output( x )
        return output
        
    def forward(self, x):
        """
        Forward pass through the BassetVL model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output

