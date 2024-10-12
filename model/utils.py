import torch
import torch.nn as nn

from model.trees.ClusTree import ClusTree
from model.trees.DDT import DDT
from model.trees.SoftDDT import SoftDDT

from config import learning_config


def get_tree(devices):
    tree = learning_config['tree']
    max_depth = learning_config['tree_max_depth']
    if tree == "ddt":
        return DDT(num_input=get_num_input(), num_output=len(devices), depth=0, max_depth=max_depth)
    elif tree == "soft-ddt":
        return SoftDDT(num_input=get_num_input(), num_output=len(devices), depth=0, max_depth=max_depth)
    elif tree == "clustree":
        return ClusTree(num_input=get_num_input(), devices=devices, depth=0, max_depth=max_depth)


def get_num_input():
    num_input = 10
    if learning_config['onehot_kind']:
        num_input = 13
    return num_input


def get_critic():
    num_input = get_num_input()

    if learning_config["learning_algorithm"] == "ppo" or learning_config["learning_algorithm"] == "a2c":
        num_hidden_layers = learning_config['critic_hidden_layer_num']
        critic_hidden_layer_dim = learning_config['critic_hidden_layer_dim']
        # Create list of layers
        layers = [nn.Linear(num_input, critic_hidden_layer_dim), nn.Sigmoid()]

        # Append hidden layers dynamically
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(critic_hidden_layer_dim, critic_hidden_layer_dim))
            layers.append(nn.Sigmoid())

        # Final output layer
        layers.append(nn.Linear(critic_hidden_layer_dim, 1))

        return nn.Sequential(*layers)
    else:
        return None
