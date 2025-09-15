import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------
# 🔨 Definizione modello ResNet18 + ArcFace
# -----------------------------
class ArcFaceLayer(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.30):
        super(ArcFaceLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.W, gain=1.0)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.W, dim=0)
        logits = torch.matmul(embeddings, W)

        if labels is not None:
            theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.m)
            one_hot = F.one_hot(labels, num_classes=logits.size(1)).float().to(logits.device)
            logits = logits * (1 - one_hot) + target_logits * one_hot

        logits = logits * self.s
        # Stabilizza softmax sottraendo il max per rimanere numericamente sicuri
        logits = logits - logits.max(dim=1, keepdim=True).values
        return logits

class ArcFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=256):
        super(ArcFaceNet, self).__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_layer.weight, gain=1.0)
        if self.embedding_layer.bias is not None:
            nn.init.zeros_(self.embedding_layer.bias)
        self.arcface = ArcFaceLayer(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = F.normalize(self.embedding_layer(features), dim=1)
        logits = self.arcface(embeddings, labels)
        return logits, embeddings