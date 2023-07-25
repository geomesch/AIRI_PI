from torch import nn
import torch


class MaskedMSELoss(nn.MSELoss):

    def forward(self, x, y, mask=None):
        if mask is not None:
            n = x.shape[0]
            loss = 0.0
            diff = (x - y) ** 2
            b = mask.dtype == torch.bool
            for diff, mask in zip(diff, mask):
                if b:
                    loss += torch.masked_select(diff, mask).mean()
                else:
                    loss += (diff * mask).mean()
            loss /= n
        else:
            shape = tuple(range(1, len(x.shape)))
            loss = ((x - y) ** 2).mean(axis=shape).mean()
        return loss
            
            
        
class PILoss(nn.Module):
    def __init__(self, spacing=(25, 50), s=2e-4):
        super().__init__()
        self.spacing = spacing
        self.s_ = s
        self.mse = MaskedMSELoss()
    
    def forward(self, p, K, mask=None):
        divk = self.calc_divk(p, K)
        p_dt = self.calc_p_dt(p)
        diff = divk - p_dt * self.s_
        q = torch.zeros_like(diff)
        q[..., 32, 32] = 1.0
        diff -= q
        return self.mse(diff ** 2, 0.0, mask=mask)
    
    def calc_divk(self, p, K):
        p_dx, p_dy = torch.gradient(p, dim=(-2, -1), edge_order=2, spacing=self.spacing)
        divk = torch.gradient(K * p_dx, dim=-2, edge_order=2, spacing=self.spacing[0])[0]\
            + torch.gradient(K * p_dy, dim=-1, edge_order=2, spacing=self.spacing[1])[0]
        return divk
    
    def calc_p_dt(self, p):
        return torch.gradient(p, dim=1, edge_order=2)[0]
