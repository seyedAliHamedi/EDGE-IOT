import torch
import numpy as np
import torch.nn as nn

from data.config import learning_config

class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, counter=0, exploration_rate=0):
        """
        Initializes the DDT structure.
        
        Args:
            num_input: Number of input features.
            num_output: Number of output classes or predictions.
            depth: Current depth of the tree.
            max_depth: Maximum depth of the tree.
            counter: Counter for exploration steps.
            exploration_rate: Initial exploration rate for the tree.
        """
        super(DDT, self).__init__()
        
        # Tree properties
        self.depth = depth
        self.max_depth = max_depth
        self.counter = counter
        
        # Exploration parameters
        self.epsilon = learning_config['explore_epsilon']
        num_epoch = learning_config['num_jobs']
        
        
        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + self.exp_mid_bound / 2
        self.exp_threshold = self.exp_mid_bound - self.exp_mid_bound / 2
        self.shouldExplore = learning_config['should_explore']  # Use boolean for clarity
        
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.ones(num_output))  # Leaf stores output probabilities
        if depth < max_depth:
            # Initialize weights, bias, and child nodes
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            # Create left and right child nodes
            self.left = DDT(num_input, num_output, depth + 1, max_depth, self.counter, self.exploration_rate)
            self.right = DDT(num_input, num_output, depth + 1, max_depth, self.counter, self.exploration_rate)

        

    def forward(self, x, path=""):
        # Leaf node: return the probability distribution
        if self.depth == self.max_depth:
            return self.prob_dist, path
        
        # Internal node: compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))
        
        # Exploration phase: adjust the value randomly
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)

        # Recursive decision: traverse left or right based on val
        if val >= 0.5:
            right_output, right_path = self.right(x, path + "R")
            return val * right_output, right_path
        else:
            left_output, left_path = self.left(x, path + "L")
            return (1 - val) * left_output, left_path
        
    def explore(self, val):
        self.counter += 1
        self.exploration_rate -= self.epsilon  # Reduce exploration rate over time
        
        # Stop exploration once the threshold is reached
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = False
        
        # Return an adjusted value for exploration
        return 1 - val
