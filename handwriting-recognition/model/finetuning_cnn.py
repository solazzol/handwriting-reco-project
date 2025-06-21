from simple_cnn import SimpleCNN
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn as nn

#fine tuning architecture
class FineTuningCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        # dropout nei blocchi intermedi
        self.model.layer1 = nn.Sequential(
            self.model.layer1,
            nn.Dropout(p=0.3)
        )
        self.model.layer2 = nn.Sequential(
            self.model.layer2,
            nn.Dropout(p=0.3)
        )

        # fully connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)