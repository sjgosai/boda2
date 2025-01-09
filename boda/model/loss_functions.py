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
import torch
import torch.nn as nn

from ..common import utils

from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss

##################
# Loss functions #
##################

class MSEKLmixed(nn.Module):
    """
    A custom loss module that combines Mean Squared Error (MSE) loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the MSE loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.MSELoss): The Mean Squared Error loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining MSE and KL divergence losses.

    Example:
        loss_fn = MSEKLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the MSEKLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining MSE and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the L1KLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining L1 and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

class MSEwithEntropy(nn.Module):
    """
    A custom loss module that combines Mean Squared Error (MSE) loss with entropy divergence loss
    based on entropy of predictions and targets.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the Mean Squared Error (MSE) loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the entropy divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the MSE loss term.
        beta (float): Scaling factor for the EDE loss term.
        MSE (nn.MSELoss): The Mean Squared Error loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining MSE and SCE losses based on entropy.

    Example:
        loss_fn = MSEwithEntropy()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the MSEwithEntropy loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the Mean Squared Error (MSE) loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the Symmetric Cross-Entropy (SCE) loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining MSE and SDE losses based on entropy.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        pred_entropy = nn.Softmax(dim=1)(preds)
        pred_entropy = torch.sum(- pred_entropy * torch.log(pred_entropy), dim=1)
        
        targ_entropy = nn.Softmax(dim=1)(targets)
        targ_entropy = torch.sum(- targ_entropy * torch.log(targ_entropy), dim=1)
        
        MSE_loss = self.MSE(preds, targets)
        EDE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        EDE_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

class L1withEntropy(nn.Module):
    """
    A custom loss module that combines L1 loss with entropy divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the entropy loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the entropy loss term.
        MSE (nn.L1Loss): The L1 loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and entropy-based losses.

    Example:
        loss_fn = L1withEntropy()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the L1withEntropy loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the entropy loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining L1 and entropy-based losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        pred_entropy = nn.Softmax(dim=1)(preds)
        pred_entropy = torch.sum(- pred_entropy * torch.log(pred_entropy), dim=1)
        
        targ_entropy = nn.Softmax(dim=1)(targets)
        targ_entropy = torch.sum(- targ_entropy * torch.log(targ_entropy), dim=1)
        
        MSE_loss = self.MSE(preds, targets)
        EDE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        EDE_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)
    
class DirichletNLLLoss(nn.Module):
    
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        
        self.reduction = reduction
        self.eps = eps
        self.activation= nn.Softplus()
        
        assert reduction in ['mean', 'sum', 'none'], "reduction must be 'mean'|'sum'|'none'"
        
    def forward(self, preds, targets):
        
        preds   = self.activation(preds)
        targets = targets.clamp(min=self.eps)
        
        term1 = torch.lgamma(preds.sum(dim=-1))
        term2 = torch.lgamma(preds).sum(dim=-1)
        term3 = torch.xlogy(preds - 1.0, targets).sum(dim=-1)

        #torch.xlogy(self.concentration - 1.0, value).sum(-1)
        #+ torch.lgamma(self.concentration.sum(-1))
        #- torch.lgamma(self.concentration).sum(-1)

        result = -term1 + term2 - term3
        
        if self.reduction == 'mean':
            result = result.mean()
        elif self.reduction == 'sum':
            result = result.sum()
        else:
            result = result
        
        return result
        
class JeffreysDivLoss(nn.Module):
    
    def __init__(self, reduction='batchmean', log_target=False, eps=1e-12):
        super().__init__()
        
        self.reduction = reduction
        self.log_target= log_target
        self.eps = eps
        self.activation= nn.LogSoftmax()
        self.KL = nn.KLDivLoss(reduction=reduction, log_target=True) # expect all inputs as log-space for symmetric use
        
    def forward(self, preds, targets):
        
        preds = self.activation(preds)
        if self.log_target:
            targets = targets.clip(torch.log(self.eps))
        else:
            targets = targets.clip(self.eps)
            targets = torch.log(targets)
        
        return self.KL(preds, targets) + self.KL(targets, preds)
        
        
L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss

def add_criterion_specific_args(parser, criterion_name):
    
    group = parser.add_argument_group('Criterion args')
    
    if criterion_name == 'L1Loss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'MSELoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'CrossEntropyLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--ignore_index', type=int, default=-100, help='Specifies a target value that is ignored and does not contribute to the input gradient. See torch.nn docs for more details.')
        group.add_argument('--label_smooting', type=float, default=0., help='A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. See torch.nn docs for more details.')
    elif criterion_name == 'CTCLoss':
        group.add_argument('--blank', type=int, default=0, help='blank label.')
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--zero_infinity', type=utils.str2bool, default=False, help='Whether to zero infinite losses and the associated gradients.')
    elif criterion_name == 'NLLLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--ignore_index', type=int, default=-100, help='Specifies a target value that is ignored and does not contribute to the input gradient. See torch.nn docs for more details.')
    elif criterion_name == 'PoissonNLLLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--log_input', type=utils.str2bool, default=True, help='See torch.nn docs for details.')
        group.add_argument('--full', type=utils.str2bool, default=False, help='whether to compute full loss, i. e. to add the Stirling approximation term')
        group.add_argument('--eps', type=float, default=1e-8, help='small value to avoid log(0) when `log_input` is False')
    elif criterion_name == 'GaussianNLLLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--full', type=utils.str2bool, default=False, help='include the constant term in the loss calculation.')
        group.add_argument('--eps', type=float, default=1e-6, help='small value to clamp.')
    elif criterion_name == 'KLDivLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--log_target', type=utils.str2bool, default=False, help='Specifies whether target is in the log space.')
    elif criterion_name == 'BCELoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'BCEWithLogitsLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'MarginRankingLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--margin', type=float, default=0.)
    elif criterion_name == 'HingeEmbeddingLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--margin', type=float, default=1.)
    elif criterion_name == 'MultiLabelMarginLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'HuberLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--delta', type=float, default=1.0, help='Specifies the threshold at which to change between delta-scaled L1 and L2 loss. The value must be positive.')
    elif criterion_name == 'SmoothL1Loss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--beta', type=float, default=1.0, help='Specifies the threshold at which to change between L1 and L2 loss. The value must be non-negative.')
    elif criterion_name == 'SoftMarginLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'MultiLabelSoftMarginLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'CosineEmbeddingLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--margin', type=float, default=0., help='Should be a number from -1 to 1, 0 to 0.5 is suggested')
    elif criterion_name == 'MultiMarginLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('-p', type=int, default=1, help="Can be 1 or 2")
        group.add_argument('--margin', type=float, default=1., help='')
    elif criterion_name == 'TripletMarginLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--margin', type=float, default=1.)
        group.add_argument('--p', type=int, default=2, help='The norm degree for pairwise distance.')
        group.add_argument('--eps', type=float, default=1e-6, help='Small constant for numerical stability.')
        group.add_argument('--swap', type=utils.str2bool, default=False, help='The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.')
    elif criterion_name == 'TripletMarginWithDistanceLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--margin', type=float, default=1., help='A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0. Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives.')
        group.add_argument('--swap', type=utils.str2bool, default=False, help='Whether to use the distance swap described in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al. If True, and if the positive example is closer to the negative example than the anchor is, swaps the positive example and the anchor in the loss computation.')
    elif criterion_name == 'MSEKLmixed':
        group.add_argument('--reduction', type=str, default='batchmean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for the MSE loss term.')
        group.add_argument('--beta', type=float, default=1.0, help='Scaling factor for the KLDIVLOSS loss term.')
    elif criterion_name == 'L1KLmixed':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for the MSE loss term.')
        group.add_argument('--beta', type=float, default=1.0, help='Scaling factor for the KLDIVLOSS loss term.')
    elif criterion_name == 'MSEwithEntropy':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for the MSE loss term.')
        group.add_argument('--beta', type=float, default=1.0, help='Scaling factor for the entropy error loss term.')
    elif criterion_name == 'L1withEntropy':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for the MSE loss term.')
        group.add_argument('--beta', type=float, default=1.0, help='Scaling factor for the entropy error loss term.')
    elif criterion_name == 'DirichletNLLLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
    elif criterion_name == 'JeffreysDivLoss':
        group.add_argument('--reduction', type=str, default='mean', help='Specifies reduction applied when loss is calculated: `"none"`|`"batchmean"`|`"mean"`|`"sum"`. See torch.nn docs for more details.')
        group.add_argument('--log_target', type=utils.str2bool, default=False, help='Specifies whether target is in the log space.')
    else:
        raise RuntimeError(f'{criterion_name} not supported. Try: [L1Loss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, MSEKLmixed, L1KLmixed, MSEwithEntropy, L1withEntropy]')
        
    return parser