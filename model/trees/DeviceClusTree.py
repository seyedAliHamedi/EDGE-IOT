import torch
import numpy as np
import torch.nn as nn

from config import learning_config
from env.utils import balance_kmeans_cluster

class DeviceClusterTree(nn.Module):
    def __init__(self, num_input, devices, depth, max_depth, counter=0, exploration_rate=0):
        """
        Initializes the DeviceClusterTree structure.

        Args:
            num_input (int): Number of input features.
            devices (list): List of devices to be clustered.
            depth (int): Current depth of the tree.
            max_depth (int): Maximum depth of the tree.
            counter (int): Counter for exploration steps.
            exploration_rate (float): Initial exploration rate for the tree.
        """
        super(DeviceClusterTree, self).__init__()

        # Tree properties
        self.depth = depth
        self.max_depth = max_depth
        self.counter = counter
        self.devices = devices  # Ensure devices are stored

        # Exploration parameters
        self.epsilon = learning_config['explore_epsilon']
        num_epoch = learning_config['num_jobs']
        
        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + self.exp_mid_bound / 2
        self.exp_threshold = self.exp_mid_bound - self.exp_mid_bound / 2
        self.shouldExplore = learning_config['should_explore']  # Use boolean for clarity
        
        if depth < max_depth:
            # Initialize weights and biases for internal nodes
            self.weights = nn.Parameter(torch.empty(num_input+learning_config['pe_num_features']*len(self.devices)).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

            # Create left and right child nodes using clustering
            clusters = balance_kmeans_cluster(self.devices)
            left_cluster = clusters[0]
            right_cluster = clusters[1]
            self.left = DeviceClusterTree(num_input, left_cluster, depth + 1, max_depth, self.counter, self.exploration_rate)
            self.right = DeviceClusterTree(num_input, right_cluster, depth + 1, max_depth, self.counter, self.exploration_rate)
        else:
            # Leaf node stores output probabilities for devices
            self.prob_dist = nn.Parameter(torch.zeros(len(devices)))

    def forward(self, x, path=""):
        # Leaf node: return the probability distribution and path
        if self.depth == self.max_depth:
            return self.prob_dist, path, self.devices

        # Internal node: compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights.t()) + self.bias))
        
        # Exploration phase: adjust the value randomly based on exploration rate
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)
        
        
        task_num_input = 8 if learning_config['onehot_kind'] else 5
        # Recursive decision: traverse left or right based on the computed value
        if val >= 0.5:
            indices = [self.devices.index(device) for device in self.right.devices]
            temp = x[task_num_input:].view(-1, learning_config['pe_num_features']) 
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[:task_num_input], temp[indices_tensor].view(-1)), dim=0)
            right_output, right_path, devices = self.right(x, path + "R")
            return val * right_output, right_path, self.right.devices
        else:
            indices = [self.devices.index(device) for device in self.left.devices]
            temp = x[task_num_input:].view(-1, learning_config['pe_num_features'])
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[:task_num_input], temp[indices_tensor].view(-1)), dim=0)
            left_output, left_path, devices = self.left(x, path + "L")
            return val * left_output, left_path, self.left.devices
        
    def explore(self, val):
        self.counter += 1
        self.exploration_rate -= self.epsilon  # Reduce exploration rate over time
        
        # Stop exploration once the threshold is reached
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = False
        
        # Return an adjusted value for exploration
        return 1 - val
