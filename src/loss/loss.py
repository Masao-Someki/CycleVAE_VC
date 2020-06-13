# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import numpy as np
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, device):
        super(Loss, self).__init__()
        self.mcd_loss = MCDLoss(device)
        self.lat_loss = LatentLoss()

    def forward(self, x, y, scaler):
        # loss between ground truth and reconst and target
        # Use MCDLoss for reconst, not L1 or MSE loss.
        # compute loss for each batch
        ret = {}
        ret['mcd/1st'] = 0
        ret['mcd/2nd'] = 0
        ret['lat/1st'] = 0
        ret['lat/2nd'] = 0
        ret['bp'] = 0

        reconst_loss = 0
        lat_loss = 0
        mcd_half = 0
        mcd_last = 0

        for c in range(len(x['reconst_half'])):
            # compute loss for each cycle and sum.
            mcd_half = self.mcd_loss(
                    x['reconst_half'][c],
                    y['feat'][..., y['cv_stats'].shape[-1]:],
                    scaler=scaler
            )
            mcd_last  = self.mcd_loss(
                    x['reconst_last'][c],
                    y['feat'][..., y['cv_stats'].shape[-1]:],
                    scaler=scaler
            )
            reconst_loss += (mcd_half + mcd_last)

            # save for log
            ret['mcd/1st'] += mcd_half.item()
            ret['mcd/2nd'] += mcd_last.item()

            # loss for latent dimension
            lat1_loss = self.lat_loss(x['latent_1'][c])
            lat2_loss = self.lat_loss(x['latent_2'][c])
            lat_loss += (lat1_loss + lat2_loss)

            # save for log
            ret['lat/1st'] += lat1_loss.item()
            ret['lat/2nd'] += lat2_loss.item()

        # take mean
        ret['bp'] = (reconst_loss + lat_loss).item()
        ret['mcd/1st'] = ret['mcd/1st'] / (c + 1)
        ret['mcd/2nd'] = ret['mcd/2nd'] / (c + 1)
        ret['lat/1st'] = ret['lat/1st'] / (c + 1)
        ret['lat/2nd'] = ret['lat/2nd'] / (c + 1)

        return reconst_loss + lat_loss, ret


class MCDLoss(nn.Module):
    def __init__(self, device):
        super(MCDLoss, self).__init__()
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')
        self.coef = 10.0 / torch.log(torch.Tensor([10]).to(device)) \
                         * torch.sqrt(torch.Tensor([2]).to(device))

    def forward(self, x, y, scaler, mse=False):
        # compute MCD loss
        l1_loss = torch.abs(x - y)
        l1_loss = l1_loss * scaler
        mcd = self.coef * torch.sum(l1_loss, 2)
        return torch.mean(mcd)

class LatentLoss(nn.Module):
    def __init__(self):
        super(LatentLoss, self).__init__()

    def forward(self, x):
        # shape of x: (B, L, D), where B: batch_size, L: length, D: latent_dim * 2
        lat_dim = x.shape[-1] // 2
        mu = x[..., :lat_dim]
        sigma = x[..., lat_dim:]

        # compute latent loss
        lat_loss = 0.5 * torch.sum(torch.exp(sigma) \
                       + torch.pow(mu, 2) - sigma - 1.0, 2) # (B, L)
        return torch.mean(lat_loss)
