import torch
import torch.nn as nn

##################
# Loss functions #
##################

class MSEKLmixed(nn.Module):
    """
    A custom loss module that combines Mean Squared Error (MSE) loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        mse_scale (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
        kl_scale (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        mse_scale (float): Scaling factor for the MSE loss term.
        kl_scale (float): Scaling factor for the KL divergence loss term.
        MSE (nn.MSELoss): The Mean Squared Error loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining MSE and KL divergence losses.

    Example:
        loss_fn = MSEKLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        """
        Initialize the MSEKLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            mse_scale (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
            kl_scale (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
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
        preds_log_prob  = preds   - preds.exp().sum(dim=1,keepdim=True).log()
        target_log_prob = targets - targets.exp().sum(dim=1,keepdim=True).log()
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        KL_loss.mul(self.kl_scale)
        
        return combined_loss.div(self.mse_scale+self.kl_scale)

class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        mse_scale (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        kl_scale (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        mse_scale (float): Scaling factor for the L1 loss term.
        kl_scale (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        """
        Initialize the L1KLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            mse_scale (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            kl_scale (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
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
        preds_log_prob  = preds   - preds.exp().sum(dim=1,keepdim=True).log()
        target_log_prob = targets - targets.exp().sum(dim=1,keepdim=True).log()
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        KL_loss.mul(self.kl_scale)
        
        return combined_loss#.div(self.mse_scale+self.kl_scale)

class MSEwithEntropy(nn.Module):
    """
    A custom loss module that combines Mean Squared Error (MSE) loss with the Symmetric Cross-Entropy (SCE) loss
    based on entropy of predictions and targets.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        mse_scale (float, optional): Scaling factor for the Mean Squared Error (MSE) loss term. Default is 1.0.
        kl_scale (float, optional): Scaling factor for the Symmetric Cross-Entropy (SCE) loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        mse_scale (float): Scaling factor for the MSE loss term.
        kl_scale (float): Scaling factor for the SCE loss term.
        MSE (nn.MSELoss): The Mean Squared Error loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining MSE and SCE losses based on entropy.

    Example:
        loss_fn = MSEwithEntropy()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        """
        Initialize the MSEwithEntropy loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            mse_scale (float, optional): Scaling factor for the Mean Squared Error (MSE) loss term. Default is 1.0.
            kl_scale (float, optional): Scaling factor for the Symmetric Cross-Entropy (SCE) loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining MSE and SCE losses based on entropy.

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
        SEE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        SEE_loss.mul(self.kl_scale)
        
        return combined_loss.div(self.mse_scale+self.kl_scale)

class L1withEntropy(nn.Module):
    """
    A custom loss module that combines L1 loss with entropy-based loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        mse_scale (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        kl_scale (float, optional): Scaling factor for the entropy loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        mse_scale (float): Scaling factor for the L1 loss term.
        kl_scale (float): Scaling factor for the entropy loss term.
        MSE (nn.L1Loss): The L1 loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and entropy-based losses.

    Example:
        loss_fn = L1withEntropy()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', mse_scale=1.0, kl_scale=1.0):
        """
        Initialize the L1withEntropy loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            mse_scale (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            kl_scale (float, optional): Scaling factor for the entropy loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.mse_scale = mse_scale
        self.kl_scale  = kl_scale
        
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
        SEE_loss = self.MSE(pred_entropy, targ_entropy)
        
        combined_loss = MSE_loss.mul(self.mse_scale) + \
                        SEE_loss.mul(self.kl_scale)
        
        return combined_loss#.div(self.mse_scale+self.kl_scale)
    
