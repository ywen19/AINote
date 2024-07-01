import torch.nn as nn
import torch

EPS = 1e-11

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


class LogCoshDiceLoss(nn.Module):
    # https://arxiv.org/pdf/2006.14822v4.pdf
    # Variant of Dice Loss and inspired regression log-cosh approach

    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        cosh = self.cal_cosh(1. - dsc)
        return torch.log(cosh+EPS)

    def cal_cosh(self, dice_loss):
        return (torch.exp(dice_loss) + torch.exp(-dice_loss))/2


class ShapeAwareLoss(nn.Module):
    def __init__(self):
        super(ShapeAwareLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        # get the euclidean distance
        dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(y_pred, y_true), 2), dim=0))/256
        # dist = torch.cdist(y_pred, y_true, p=2.0)
        # get the cross entropy 
        ce = nn.functional.binary_cross_entropy(y_pred, y_true)
        # calculate the loss
        loss = ce + dist*ce
        return loss




