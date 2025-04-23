# -*- coding: utf-8 -*-
"""
Implements the central server for federated learning.
"""

import torch

class Server:
    def __init__(self, global_model):
        self.global_model = global_model
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)

    def aggregate_models(self):
        global_params = self.global_model.state_dict()
        client_params = [client.get_model_params() for client in self.clients]
        num_clients = len(self.clients)

        with torch.no_grad():
            for key in global_params.keys():
                global_params[key] = torch.stack([client_params[i][key].float() for i in range(num_clients)], dim=0).mean(dim=0)

        self.global_model.load_state_dict(global_params)

    def distribute_model(self):
        global_params = self.global_model.state_dict()
        for client in self.clients:
            client.load_model_params(global_params)

    def evaluate_global_model(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f'Global Model Accuracy on test data: {accuracy:.2f}%')
        return accuracy