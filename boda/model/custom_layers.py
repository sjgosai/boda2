"""
MIT License

Copyright (c) 2025 Sagar Gosai, Rodrigo Castro

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
import math

from collections import OrderedDict

import torch
import torch.nn as nn

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

class GroupedLinear(nn.Module):
    """
    A custom linear transformation module that groups input and output features.

    Args:
        in_group_size (int): Number of input features in each group.
        out_group_size (int): Number of output features in each group.
        groups (int): Number of groups.

    Attributes:
        in_group_size (int): Number of input features in each group.
        out_group_size (int): Number of output features in each group.
        groups (int): Number of groups.
        weight (Parameter): Learnable weight parameter for the linear transformation.
        bias (Parameter): Learnable bias parameter for the linear transformation.

    Methods:
        reset_parameters(weights, bias):
            Initialize the weight and bias parameters with kaiming uniform initialization.
        forward(x):
            Apply the grouped linear transformation to the input tensor.

    Example:
        linear_layer = GroupedLinear(in_group_size=10, out_group_size=5, groups=2)
        output = linear_layer(input_tensor)
    """
    
    def __init__(self, in_group_size, out_group_size, groups):
        """
        Initialize the GroupedLinear module.

        Args:
            in_group_size (int): Number of input features in each group.
            out_group_size (int): Number of output features in each group.
            groups (int): Number of groups.

        Returns:
            None
        """
        super().__init__()
        
        self.in_group_size = in_group_size
        self.out_group_size= out_group_size
        self.groups        = groups
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias   = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        """
        Initialize the weight and bias parameters with kaiming uniform initialization.

        Args:
            weights (Tensor): The weight parameter tensor.
            bias (Tensor): The bias parameter tensor.

        Returns:
            None
        """
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        """
        Apply the grouped linear transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed output tensor.
        """
        reorg = x.permute(1,0).reshape(self.groups, self.in_group_size, -1).permute(0,2,1)
        hook  = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0,2,1).reshape(self.out_group_size*self.groups,-1).permute(1,0)
        
        return reorg

class RepeatLayer(nn.Module):
    """
    A custom module to repeat the input tensor along specified dimensions.

    Args:
        *args (int): Size of repetitions along each specified dimension.

    Attributes:
        args (tuple): Sizes of repetitions along each specified dimension.

    Methods:
        forward(x):
            Repeat the input tensor along the specified dimensions.

    Example:
        repeat_layer = RepeatLayer(2, 3)
        output = repeat_layer(input_tensor)
    """
    
    def __init__(self, *args):
        """
        Initialize the RepeatLayer module.

        Args:
            *args (int): Size of repetitions along each specified dimension.

        Returns:
            None
        """
        super().__init__()
        self.args = args
        
    def forward(self, x):
        """
        Repeat the input tensor along the specified dimensions.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The repeated output tensor.
        """
        return x.repeat(*self.args)
    
class BranchedLinear(nn.Module):
    """
    A custom module that implements a branched linear architecture.

    Args:
        in_features (int): Number of input features.
        hidden_group_size (int): Number of hidden features in each group.
        out_group_size (int): Number of output features in each group.
        n_branches (int): Number of branches.
        n_layers (int): Number of layers in each branch.
        activation (str): Activation function to use in the hidden layers.
        dropout_p (float): Dropout probability applied to hidden layers.

    Attributes:
        in_features (int): Number of input features.
        hidden_group_size (int): Number of hidden features in each group.
        out_group_size (int): Number of output features in each group.
        n_branches (int): Number of branches.
        n_layers (int): Number of layers in each branch.
        branches (OrderedDict): Dictionary to store branch layers.
        nonlin (nn.Module): Activation function module.
        dropout (nn.Dropout): Dropout layer module.
        intake (RepeatLayer): A layer to repeat input along branches.

    Methods:
        forward(x):
            Perform forward pass through the branched linear architecture.

    Example:
        branched_linear = BranchedLinear(in_features=256, hidden_group_size=128,
                                         out_group_size=64, n_branches=4,
                                         n_layers=3, activation='ReLU', dropout_p=0.5)
        output = branched_linear(input_tensor)
    """
    
    def __init__(self, in_features, hidden_group_size, out_group_size, 
                 n_branches=1, n_layers=1, 
                 activation='ReLU', dropout_p=0.5):
        """
        Initialize the BranchedLinear module.

        Args:
            in_features (int): Number of input features.
            hidden_group_size (int): Number of hidden features in each group.
            out_group_size (int): Number of output features in each group.
            n_branches (int): Number of branches.
            n_layers (int): Number of layers in each branch.
            activation (str): Activation function to use in the hidden layers.
            dropout_p (float): Dropout probability applied to hidden layers.

        Returns:
            None
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers   = n_layers
        
        self.branches = OrderedDict()
        
        self.nonlin  = getattr(nn, activation)()                               
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        
        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size
            
    def forward(self, x):
        """
        Perform forward pass through the branched linear architecture.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        hook = self.intake(x)
        
        i = -1
        for i in range(self.n_layers-1):
            hook = getattr(self, f'branched_layer_{i+1}')(hook)
            hook = self.dropout( self.nonlin(hook) )
        hook = getattr(self, f'branched_layer_{i+2}')(hook)
            
        return hook
    
