import torch
import torch.nn as nn


class Loss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, hyp, device):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        """
        super().__init__()

        self.nc = hyp['num_classes']

        da_w = torch.ones(self.nc)
        da_w[0] = 0.02

        
        self.losses = (nn.CrossEntropyLoss(weight=da_w)).to(device)
        


    def forward(self, predictions, targets):
        """

        Args:
            predictions: predicts of result
            targets: GT

        Returns:
            loss: loss
        """
        return self.losses(predictions, targets)