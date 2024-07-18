import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class Vit(nn.Module):
    def __init__(self, hash_bit):
        super(Vit, self).__init__()
        model_vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.vitbone=create_feature_extractor(model_vit, return_nodes=['getitem_5'])
        self.hash_layer= nn.Sequential(nn.Linear(768, hash_bit),nn.Tanh())
    def forward(self, x):
        x = self.vitbone(x)['getitem_5']
        y = self.hash_layer(x)
        return x,y