import numpy as np
import torch
import torch.nn as nn

from model.utils import balance_kmeans_cluster

class ClusterTree(nn.Module):
    def __init__(self, devices, depth, max_depth):
        super(ClusterTree, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        self.devices = devices
        # 5 weights for task and 4 for each device
        num_features = 5 + 4 * len(devices)
        
        self.exploration_rate=0.5
        self.explore_decay=0.99

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_features).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(len(devices)))

        if depth < max_depth:
            clusters = balance_kmeans_cluster(self.devices)
            left_cluster = clusters[0]
            right_cluster = clusters[1]
            self.left = ClusterTree(left_cluster, depth+1, max_depth)
            self.right = ClusterTree(right_cluster, depth+1, max_depth)

    def forward(self, x,path=""):
        if self.depth == self.max_depth:
            return self.prob_dist,path,self.devices

        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))
        
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)
            
        if val >= 0.5:
            indices = [self.devices.index(device)
                       for device in self.right.devices]
            temp = x[5:].view(-1, 4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            right_output, right_path ,devices= self.right(x, path + "R")
            return val * right_output, right_path,self.right.devices
        else:
            indices = [self.devices.index(device)
                       for device in self.left.devices]
            temp = x[5:].view(-1, 4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            left_output, left_path,devices = self.left(x, path + "L")
            return val * left_output, left_path,self.left.devices
        
        
    def explore(self,val):
        self.counter += 1
        self.exploration_rate -= self.epsilon
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = 0
        return (1 - val)
    