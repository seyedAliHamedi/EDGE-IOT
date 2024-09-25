
import torch
import torch.nn as nn

from data.db import Database
from model.trees.ClusTree import ClusTree
from model.trees.DDT import DDT
from model.trees.DeviceClusTree import DeviceClusterTree
from model.trees.SoftDDT import SoftDDT
        
from config import learning_config

def get_tree():
    tree = learning_config['tree']
    devices = Database().get_all_devices()
    max_depth = learning_config['tree_max_depth']
    if tree == "ddt":
        return DDT(num_input=get_num_input(devices=False),num_output=len(devices),depth=0,max_depth=max_depth)
    elif tree == "device-ddt":
        return DDT(num_input=get_num_input(devices=True),num_output=len(devices),depth=0,max_depth=max_depth)
    elif tree == "soft-ddt":
        return SoftDDT(num_input=get_num_input(devices=False),num_output=len(devices),depth=0,max_depth=max_depth)
    elif tree == "soft-device-ddt":
        return SoftDDT(num_input=get_num_input(devices=True),num_output=len(devices),depth=0,max_depth=max_depth)
    elif tree == "clustree":
        return ClusTree(num_input=get_num_input(devices=False),devices=devices,depth=0,max_depth=max_depth)
    elif tree == "device-clustree":
        return DeviceClusterTree(num_input=get_num_input(devices=False),devices=devices,depth=0,max_depth=max_depth)


def get_num_input(devices):
    if devices:
        pe_feature = learning_config['pe_num_features'] * len(Database().get_all_devices())
        num_input = 5 + pe_feature
        if learning_config['onehot_kind']:
            num_input = 8  + pe_feature
        return num_input
    else:
        num_input = 5
        if learning_config['onehot_kind']:
            num_input = 8
        return num_input
    
def get_critic():
    tree = learning_config['tree']
    num_input =get_num_input(devices=False)
    if tree in ("device-clustree","soft-device-ddt","device-ddt"):
        num_input =get_num_input(devices=True)
        
    if learning_config["learning_algorithm"] == "ppo" or learning_config["learning_algorithm"] == "a2c" :
        num_hidden_layers = learning_config['critic_hidden_layer_num']
        critic_hidden_layer_dim = learning_config['critic_hidden_layer_dim']
        # Create list of layers
        layers = [nn.Linear(num_input, critic_hidden_layer_dim), nn.ReLU()]

        # Append hidden layers dynamically
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(critic_hidden_layer_dim, critic_hidden_layer_dim))
            layers.append(nn.ReLU())

        # Final output layer
        layers.append(nn.Linear(critic_hidden_layer_dim, 1))

        return  nn.Sequential(*layers)
    else:
        return None
        