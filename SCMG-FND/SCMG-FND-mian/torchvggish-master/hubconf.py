import torch
from torch import nn
import numpy as np

dependencies = ['torch', 'numpy']

def vggish(pretrained=True, **kwargs):
    """
    VGGish model for audio feature extraction
    pretrained (bool): If True, returns a model pre-trained on AudioSet
    """
    model = VGGish(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
            progress=True)
        model.load_state_dict(state_dict)
    return model

class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torchvggish
    """
    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1)
        x = self.embeddings(x)
        return x 