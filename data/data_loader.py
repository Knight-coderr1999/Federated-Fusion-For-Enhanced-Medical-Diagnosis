# -*- coding: utf-8 -*-
"""
Handles loading and preprocessing of multi-modal medical image data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os

class MultiModalDataset(Dataset):
    def __init__(self, data_paths, modalities, transform=None):
        """
        Args:
            data_paths (dict): Dictionary of paths to data for each modality.
                               Example: {'mri': [...], 'pet': [...], 'ct': [...]}
            modalities (list): List of modalities to include.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.data_paths = data_paths
        self.modalities = modalities
        self.transform = transform
        self.samples = self._create_samples()

    def _create_samples(self):
        # Logic to pair corresponding images from different modalities
        samples = []
        if 'mri' in self.modalities and 'pet' in self.modalities:
            num_mri = len(self.data_paths['mri'])
            num_pet = len(self.data_paths['pet'])
            for i in range(min(num_mri, num_pet)):
                samples.append({'mri': self.data_paths['mri'][i], 'pet': self.data_paths['pet'][i]})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}
        for modality, path in sample.items():
            # Load the image data using appropriate libraries (e.g., nibabel.load(path).get_fdata())
            # For simplicity, let's assume a placeholder loading function
            image = self._load_image(path)
            if self.transform:
                image = self.transform(image)
            data[modality] = image

        # Return the multi-modal data and potentially a label
        # Example: return {'mri': mri_img, 'pet': pet_img}, label
        return data, 0 # Placeholder label

    def _load_image(self, path):
        # Placeholder function to simulate loading an image
        # Replace with your actual image loading logic
        return np.random.rand(64, 64) # Example 2D image

def get_data_loaders(data_dir, modalities, batch_size=32):
    """
    Returns PyTorch DataLoaders for training and validation.
    """
    # Define paths to your data within the data_dir
    data_paths = {
        'mri': [os.path.join(data_dir, 'mri', f'mri_{i}.nii.gz') for i in range(100)], # Example paths
        'pet': [os.path.join(data_dir, 'pet', f'pet_{i}.nii.gz') for i in range(100)],
        'ct': [os.path.join(data_dir, 'ct', f'ct_{i}.nii.gz') for i in range(100)],
    }

    transform = transforms.Compose()

    train_dataset = MultiModalDataset(data_paths, modalities, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a similar validation dataset and loader
    val_dataset = MultiModalDataset(data_paths, modalities, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader