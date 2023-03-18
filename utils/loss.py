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

        # drivable area segmentation loss criteria
        da_w = torch.Tensor([0.02,1])

        # self.losses = (nn.CrossEntropyLoss(weight=da_w) if self.nc > 2 else  \
        #         nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['seg_pos_weight']]))).to(device)
        
        self.losses = (nn.CrossEntropyLoss(weight=da_w)).to(device)
                # nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['seg_pos_weight']]))).to(device)
        


    def forward(self, predictions, targets):
        """

        Args:
            predictions: predicts of result
            targets: GT

        Returns:
            loss: loss
        """
        daseg = self.losses

        # if(self.nc == 2):
        #     loss = daseg(predictions.view(-1), targets.view(-1))
        # else:
        loss = daseg(predictions, targets)
        return loss