import torch
import torch.nn as nn
import torch.optim as optim

from data.config import learning_config
from model.utils import get_num_input

class DDTForest(nn.Module):
    def __init__(self, devices):
        """
        Initializes the DDTForest which manages a forest of decision trees (SubDDTs).

        Args:
            subtree_input_dims (int): Number of input dimensions for each subtree.
            subtree_max_depth (int): Maximum depth of each subtree.
            devices (list): List of devices to initialize trees for.
            subtree_lr (float): Learning rate for the subtree optimizers.
        """
        super(DDTForest, self).__init__()

        # Initialize the decision trees for each device
        self.forest = [
            self.create_tree(get_num_input(devices=False) + device['num_cores'], device['num_cores'] * 3)
            for device in devices
        ]
        
        # Initialize optimizers for each tree in the forest
        self.optimizers = [optim.Adam(tree.parameters(), lr=learning_config['subtree_lr']) for tree in self.forest]

    def create_tree(self, input_dims, output_dim):
        return SubDDT(input_dims, output_dim, 0, learning_config["subtree_max_depth"])


class SubDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        """
        Initializes the SubDDT structure.

        Args:
            num_input (int): Number of input features.
            num_output (int): Number of output classes or probabilities.
            depth (int): Current depth of the tree.
            max_depth (int): Maximum depth of the tree.
        """
        super(SubDDT, self).__init__()

        self.depth = depth
        self.max_depth = max_depth

        # Decision node initialization
        if depth < max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

            # Initialize left and right subtrees
            self.left = SubDDT(num_input, num_output, depth + 1, max_depth)
            self.right = SubDDT(num_input, num_output, depth + 1, max_depth)
        else:
            # Leaf node stores output probabilities for the output classes
            self.prob_dist = nn.Parameter(torch.zeros(num_output))

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist  # Return policy at leaf node

        # Compute decision value using weights and bias
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)

        # Recursive call to left or right subtree based on the decision value
        if val >= 0.5:
            return val * self.right(x)  # Traverse right subtree
        else:
            return (1 - val) * self.left(x)  # Traverse left subtree
