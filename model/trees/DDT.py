import numpy as np
import torch
import torch.nn as nn
from data.db import Database

class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, num_epoch=len(Database().get_all_jobs()), counter=0, exploration_rate=0):
        super(DDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.epsilon = 1e-5
        
        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + (self.exp_mid_bound / 2)
        self.exp_threshold = self.exp_mid_bound - (self.exp_mid_bound / 2)
        self.shouldExplore = 0
        
        self.counter = counter 
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.ones(num_output))
        if depth < max_depth:
            self.left = DDT(num_input, num_output, depth + 1, max_depth,self.counter, self.exploration_rate)
            self.right = DDT(num_input, num_output, depth + 1, max_depth,self.counter, self.exploration_rate)


    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
        
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))
        
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)

        if val >= 0.5:
            right_output, right_path = self.right(x, path + "R")
            return val * right_output, right_path
        else:
            left_output, left_path = self.left(x, path + "L")
            return (1 - val) * left_output, left_path
        
    def explore(self,val):
        self.counter += 1
        self.exploration_rate -= self.epsilon
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = 0
        return (1 - val)
    
    