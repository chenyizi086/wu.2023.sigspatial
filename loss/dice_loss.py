import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):    
        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true, axis = (0,2,3))
        union = torch.sum(y_pred, axis=(0,2,3)) + torch.sum(y_true, axis=(0,2,3))

        # Calculate Dice score and return loss
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_score.mean()

        return loss
