# -*- coding: utf-8 -*-
"""
Simulates a federated learning client.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model, local_data, learning_rate=0.01):
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.local_data):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f'Client {self.client_id} Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    def get_model_params(self):
        return self.model.state_dict()

    def load_model_params(self, params):
        self.model.load_state_dict(params)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f'Client {self.client_id} Accuracy on test data: {accuracy:.2f}%')
        return accuracy