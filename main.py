# federated_fusion_project/main.py

"""
Main script to run the federated learning simulation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from federated_fusion_project.data.data_loader import get_data_loaders
from federated_fusion_project.models.fusion_model import MultiModalFusionModel
from federated_fusion_project.clients.client import Client
from federated_fusion_project.server.server import Server
import numpy as np

# Hyperparameters
num_clients = 3
num_epochs = 5
batch_size = 16
learning_rate = 0.01
modalities = ['mri', 'pet', 'ct'] # Example modalities
input_channels = {'mri': 1, 'pet': 1, 'ct': 1} # Adjust based on your data
hidden_channels = 16
num_classes = 2 # Example: binary classification

# Simulate data (replace with your actual data loading)
def create_dummy_data(num_samples, modalities):
    data = {}
    for mod in modalities:
        data[mod] = torch.randn(num_samples, 1, 64, 64)
    labels = torch.randint(0, num_classes, (num_samples,))
    return data, labels

if __name__ == "__main__":
    # Create a global model
    global_model = MultiModalFusionModel(modalities, input_channels, hidden_channels, num_classes)

    # Initialize the server
    server = Server(global_model)

    # Create dummy data and data loaders for each client
    client_data_loaders = []
    for i in range(num_clients):
        num_samples = 100
        client_modalities = np.random.choice(modalities, size=np.random.randint(1, len(modalities) + 1), replace=False).tolist()
        client_data, client_labels = create_dummy_data(num_samples, client_modalities)
        # Prepare data for DataLoader (you might need to adjust this based on your MultiModalDataset)
        client_dataset = TensorDataset(*[client_data.get(mod, torch.zeros(num_samples, 1, 64, 64)) for mod in modalities], client_labels)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client = Client(i, MultiModalFusionModel(modalities, input_channels, hidden_channels, num_classes), client_loader, learning_rate)
        server.add_client(client)
        client_data_loaders.append(client_loader)

    # Create a dummy test loader for global evaluation
    test_data, test_labels = create_dummy_data(50, modalities)
    test_dataset = TensorDataset(*[test_data.get(mod, torch.zeros(50, 1, 64, 64)) for mod in modalities], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Federated learning rounds
    for round_num in range(num_epochs):
        print(f"\n--- Round {round_num + 1} ---")

        # Distribute global model to clients
        server.distribute_model()

        # Train each client locally
        for client in server.clients:
            client.train(epochs=1) # Train for 1 epoch in each round

        # Aggregate model updates
        server.aggregate_models()

        # Evaluate the global model
        server.evaluate_global_model(test_loader)

    print("\nFederated learning finished!")