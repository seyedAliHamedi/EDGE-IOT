import numpy as np
import torch
import torch.nn as nn

from model.utils import balance_kmeans_cluster

class ClusTree(nn.Module):
    def __init__(self, devices, depth, max_depth,num_epoch, counter=0):
        super(ClusTree, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        self.epsilon = 1e-5

        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + (self.exp_mid_bound / 2)
        self.exp_threshold = self.exp_mid_bound - (self.exp_mid_bound / 2)
        self.shouldExplore = 1
        self.counter = counter

        self.devices = devices
        num_features = 8
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_features).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.ones(len(devices)))

        if depth < max_depth:
            clusters = balance_kmeans_cluster(self.devices)
            left_cluster = clusters[0]
            right_cluster = clusters[1]
            self.left = ClusTree(left_cluster, depth+1, max_depth)
            self.right = ClusTree(right_cluster, depth+1, max_depth)

    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path, self.devices

        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)
            
        if val >= 0.5:
            right_output, right_path, devices = self.right(x, path + "R")
            return val * right_output, right_path, devices
        else:
            left_output, left_path, devices = self.left(x, path + "L")
            return (1 - val) * left_output, left_path, devices
        
        
    def explore(self,val):
        self.counter += 1
        self.exploration_rate -= self.epsilon
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = 0
        return (1 - val)
