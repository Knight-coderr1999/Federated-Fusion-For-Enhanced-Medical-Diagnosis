# -*- coding: utf-8 -*-
"""
Defines the neural network architecture for multi-modal image fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(ModalityEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Add more layers as needed

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        return x

class FusionModule(nn.Module):
    def __init__(self, num_modalities, hidden_channels):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(num_modalities * hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Add attention mechanisms or other fusion layers

    def forward(self, features):
        # Concatenate features from different modalities
        fused_features = torch.cat(features, dim=1)
        fused_features = self.relu(self.conv1(fused_features))
        return fused_features

class MultiModalFusionModel(nn.Module):
    def __init__(self, modalities, input_channels, hidden_channels, num_classes):
        super(MultiModalFusionModel, self).__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict({
            modality: ModalityEncoder(input_channels[modality], hidden_channels)
            for modality in modalities
        })
        self.fusion = FusionModule(len(modalities), hidden_channels)
        self.classifier = nn.Linear(hidden_channels * 8 * 8, num_classes) # Adjust based on feature map size

    def forward(self, input_data):
        modality_features = []
        for modality in self.modalities:
            if modality in input_data:
                modality_features.append(self.encoders[modality](input_data[modality]))
            else:
                # Handle missing modalities (e.g., with a zero tensor)
                modality_features.append(torch.zeros_like(modality_features)) # Placeholder

        fused_features = self.fusion(modality_features)
        fused_features = fused_features.view(fused_features.size(0), -1) # Flatten
        output = self.classifier(fused_features)
        return output