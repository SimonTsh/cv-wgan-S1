import torch
import torch.nn as nn

class AbsBCEWithLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, z, label):
        return self.loss(torch.abs(z), label)

    def accuracy(self, preds_real, preds_fake):
        return (torch.abs(preds_real) > 0.5).sum().item() + (torch.abs(preds_fake) < 0.5).sum().item()

class MeanBCEWithLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, z, label):
        return (self.loss(z.real, label) + self.loss(z.imag, label))*0.5

    def accuracy(self, preds_real, preds_fake):
        mean = lambda x, y: (x+y)/2
        return (mean(preds_real.real, preds_real.imag) > 0.5).sum().item() + (mean(preds_fake.real, preds_fake.imag) < 0.5).sum().item()

class MeanComplexLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return torch.view_as_real(z).mean(dim=-1)

class AbsComplexLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return z.abs()

def get_loss(loss_name):
    if loss_name != "" and loss_name != None:
        return eval(f"{loss_name}()")