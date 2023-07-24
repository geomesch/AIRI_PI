from torch import nn
import torch


class MaskedMSELoss(nn.MSELoss):

    def forward(self, x, y, mask=None):
        if mask is not None:
            n = x.shape[0]
            loss = 0.0
            diff = (x - y) ** 2
            for diff, mask in zip(diff, mask):
                loss += torch.masked_select(diff, mask).mean()
            loss /= n
        else:
            shape = tuple(range(1, len(x.shape)))
            loss = ((x - y) ** 2).mean(axis=shape).mean()
        return loss
            
            
        
        