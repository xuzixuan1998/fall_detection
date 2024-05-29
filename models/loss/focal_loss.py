import torch
import torch.nn as nn

import pdb
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for binary or multi-class classification.

        Args:
            gamma (float): Focusing parameter. Default is 2.0.
            alpha (float or list of floats): Balancing factor. Default is None.
            reduction (str): Reduction method ('none', 'mean', 'sum'). Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        else:
            self.alpha = None

    def forward(self, inputs, targets):
            """
            Forward pass of the loss function.

            Args:
                inputs (torch.Tensor): Predictions from the model (probabilities).
                targets (torch.Tensor): Ground truth labels.
            
            Returns:
                torch.Tensor: Computed focal loss.
            """
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
                inputs = inputs.transpose(1, 2)
                inputs = inputs.contiguous().view(-1, inputs.size(2))

            pt = inputs[targets.bool()]
            log_pt = pt.log()

            if self.alpha is not None:
                if self.alpha.type() != inputs.type():
                    self.alpha = self.alpha.type_as(inputs)
                alpha_tensor = self.alpha.unsqueeze(0).expand_as(targets)
                at = (alpha_tensor * targets).sum(dim=1)
                log_pt = log_pt * at

            loss = -1 * (1 - pt) ** self.gamma * log_pt

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

# Example usage:
if __name__ == "__main__":
    criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
    inputs = torch.randn(3, 5, requires_grad=True)  # Example model outputs (logits)
    targets = torch.tensor([1, 0, 4])  # Example ground truth labels
    loss = criterion(inputs, targets)
    print(f"Computed Focal Loss: {loss.item()}")