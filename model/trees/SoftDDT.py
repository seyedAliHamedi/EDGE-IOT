import torch
import torch.nn as nn
class SoftDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth ):
        super(SoftDDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.ones(num_output))
        if depth < max_depth:
            self.left = SoftDDT(num_input, num_output, depth + 1, max_depth)
            self.right = SoftDDT(num_input, num_output, depth + 1, max_depth)


    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
        
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))

        # Recursive call to left and right subtrees
        left_output, left_path = self.left(x, path + "L")
        right_output, right_path = self.right(x, path + "R")

        # Combine outputs and paths based on decision value
        output = val * right_output + (1 - val) * left_output
        final_path = right_path if val >= 0.5 else left_path
        
        return output, final_path