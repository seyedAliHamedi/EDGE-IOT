import torch
import torch.nn as nn
import torch.optim as optim
# Core Scheduler manages a forest of decision trees (SubDDTs)
class DDTForest(nn.Module):
    def __init__(self, subtree_input_dims, subtree_max_depth, devices, subtree_lr=0.005):
        super(DDTForest, self).__init__()
        
        # Initialize the decision trees for each device
        self.forest = [self.create_tree(subtree_input_dims + device['num_cores'], subtree_max_depth, device['num_cores']*3) for device in devices]
        self.optimizers = [optim.Adam(tree.parameters(), lr=subtree_lr) for tree in self.forest]

    # Create a decision tree (SubDDT) for the scheduler
    def create_tree(self, input_dims, subtree_max_depth, output_dim):
        return SubDDT(input_dims, output_dim, 0, subtree_max_depth)


# SubDDT (used by DDTForest) is a simplified version of the DDT
class SubDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(SubDDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        # Decision node
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

        # Leaf node
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))

        # Left and right subtrees
        if depth < max_depth:
            self.left = SubDDT(num_input, num_output, depth + 1, max_depth)
            self.right = SubDDT(num_input, num_output, depth + 1, max_depth)

    # Forward pass through SubDDT
    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist  # Return policy at leaf node

        # Compute decision value
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)

        # Recursive call to left or right subtree based on the decision value
        if val >= 0.5:
            return val * self.right(x)
        else:
            return (1 - val) * self.left(x)
