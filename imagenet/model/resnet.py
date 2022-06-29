import torch.nn as nn
import torchvision.models as models


class SupCEResNet(nn.Module):
    """official Resnet for image classification, e.g., ImageNet"""
    def __init__(self, name='resnet50'):
        super(SupCEResNet, self).__init__()
        self.encoder = models.__dict__[name](pretrained=True)
        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.fc(self.encoder(x))
