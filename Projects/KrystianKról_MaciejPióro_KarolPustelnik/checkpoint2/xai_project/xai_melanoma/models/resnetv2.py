import torch
from torchvision.models import resnet50, resnet101





class ResNet101(torch.nn.Module):
    def __init__(self, out_features=3, freeze = False, in_channels=3):
        super().__init__()
        self.out_features = out_features
        self.in_channels = in_channels
        self.backbone = resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
        
        self.backbone.conv1 = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=64,
                                        kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.fc = torch.nn.Linear(in_features=2048, out_features=self.out_features, bias=True)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    
    
class ResNet50(torch.nn.Module):
    def __init__(self, out_features = 3, freeze = False, in_channels=3):
        super().__init__()
        
        self.out_features = out_features
        self.in_channels = in_channels
        self.backbone = resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        self.backbone.conv1 = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=64,
                                              kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
        self.backbone.fc = torch.nn.Linear(in_features=2048, out_features=self.out_features, bias=True)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        x = self.backbone(x)
        return x