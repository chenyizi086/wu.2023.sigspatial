import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        num = pred.size(0)
        m1 = pred.contiguous().view(num, -1)
        m2 = target.contiguous().view(num, -1)
        intersection = (m1 * m2).sum()
    
        return 1-(2. * intersection + self.smooth) / (m1.sum() + m2.sum() + self.smooth)
