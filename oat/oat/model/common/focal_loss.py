"""
Focal Loss implementation for handling class imbalance and focusing on hard examples.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.
    
    Focal loss applies a modulating term to the cross entropy loss to focus learning
    on hard misclassified examples. It down-weights the loss assigned to well-classified
    examples (high probability) and focuses on hard, misclassified examples.
    
    Args:
        gamma (float): Focusing parameter for modulating loss. Higher values increase
            the effect of down-weighting easy examples. Typical values: [0, 5].
            - gamma=0: equivalent to standard cross-entropy
            - gamma=2: commonly used default (from paper)
            - gamma>2: even more focus on hard examples
        alpha (float or None): Balancing parameter for class importance. 
            - If None: no class balancing
            - If float: applies same alpha to all classes as α * (1-p_t)^γ * log(p_t)
            - For multi-class, can be extended to per-class weights
        reduction (str): Specifies the reduction to apply to the output:
            - 'none': no reduction
            - 'mean': mean of the output
            - 'sum': sum of the output
        ignore_index (int): Specifies a target value that is ignored and does not
            contribute to the input gradient. Default: -100 (same as F.cross_entropy)
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: (N,) where each value is in [0, C-1]
        - Output: scalar if reduction is 'mean' or 'sum', otherwise (N,)
    
    Examples:
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> logits = torch.randn(32, 100)  # batch_size=32, num_classes=100
        >>> targets = torch.randint(0, 100, (32,))
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits of shape (N, C) or (N*T, C)
            targets: Ground truth labels of shape (N,) or (N*T,)
        
        Returns:
            Focal loss scalar or tensor depending on reduction
        """
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probability of the target class for each example
        # targets: (N,), log_probs: (N, C)
        # We need to gather the log_prob corresponding to the target class
        targets_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Compute probabilities (p_t)
        probs = torch.exp(targets_log_probs)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Compute focal loss
        focal_loss = -focal_weight * targets_log_probs
        
        # Apply alpha balancing if specified
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
            
            # Adjust reduction to account for ignored elements
            if self.reduction == 'mean':
                return focal_loss.sum() / mask.float().sum().clamp(min=1.0)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'gamma={self.gamma}, '
            f'alpha={self.alpha}, '
            f'reduction={self.reduction}, '
            f'ignore_index={self.ignore_index})'
        )


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = None,
    reduction: str = 'mean',
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Functional interface for focal loss.
    
    Args:
        logits: Predicted logits of shape (N, C)
        targets: Ground truth labels of shape (N,)
        gamma: Focusing parameter
        alpha: Balancing parameter
        reduction: 'none', 'mean', or 'sum'
        ignore_index: Target value to ignore
    
    Returns:
        Focal loss
    """
    return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, ignore_index=ignore_index)(logits, targets)
