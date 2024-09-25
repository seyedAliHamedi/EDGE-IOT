import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import learning_config
from data.db import Database
from model.trees.ClusTree import ClusTree
from model.trees.DeviceClusTree import DeviceClusterTree
from model.utils import get_tree, get_critic
from utils import extract_pe_data

class ActorCritic(nn.Module):
    def __init__(self,logit_regressor):
        super(ActorCritic, self).__init__()
        self.discount_factor = learning_config['discount_factor']
        self.actor = get_tree()  # Initialize actor
        self.critic = get_critic()  # Critic could be None
        
        self.reset_memory()
        self.old_log_probs = {i: None for i in range(len(Database().get_all_devices()))} 
        
        self.logit_regressor = logit_regressor


    def add_new_device(self,new_device):
        self.actor.add_device(new_device, logit_regressor=self.logit_regressor) 
        self.old_log_probs[len(Database().get_all_devices())-1]=None
        
    def remove_new_device(self, device_index):
        # Remove the device from actor
        self.actor.remove_device(device_index)
        
        # Remove the specific device log prob and shift the rest
        if device_index in self.old_log_probs:
            del self.old_log_probs[device_index]
        
        # Shift remaining keys to the left
        updated_log_probs = {}
        current_index = 0  # Start from 0 for shifted index
        for key in sorted(self.old_log_probs.keys()):
            if key > device_index:
                # Only shift keys greater than the removed device
                updated_log_probs[current_index] = self.old_log_probs[key]
                current_index += 1
            else:
                updated_log_probs[key] = self.old_log_probs[key]
        
        # Replace the old log probs with the updated one
        self.old_log_probs = updated_log_probs

        
        
        
    # Store experiences in memory
    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # Clear memory after an episode
    def reset_memory(self):
        self.rewards, self.actions, self.states, self.pis = [], [], [], []

    # Forward pass through both actor and critic (if present)
    def forward(self, x):
        # Get policy distribution and path from the actor
        # Determine if the actor is ClusTree or DDT based on its output
        if isinstance(self.actor, ClusTree) or isinstance(self.actor, DeviceClusterTree)  :
            p, path,devices = self.actor(x)  # Get policy distribution and path from the actor
        else:
            p, path = self.actor(x)  # Get policy distribution and path from the actor
            devices = None  # tree does not return devices
            
  
        v = self.critic(x) if self.critic is not None else None  # Value estimate from the critic, if applicable
        return p, path, devices, v

    # Select action based on actor's policy
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        pi, path, devices, _ = self.forward(state)
        
        self.pis.append(pi)  # Store policy distribution

        probs = F.softmax(pi, dim=-1)
        
        dist = Categorical(probs)  # Create a categorical distribution over actions
        action = dist.sample()
        
        # # Reshape the feature data if needed (assuming pe_data is a 1D array)
        # pe_data = np.array(extract_pe_data(Database().get_device(action.item()))).reshape(1, -1)
        # # Reshape the target y (pi[action.item()].detach()) into a 1D array
        # target = np.array([pi[action.item()].detach()])
        # # Fit the regressor
        # self.logit_regressor.fit(pe_data, target)
        
        return action.item(), path, devices  # Return sampled action, the path, and devices

    # Calculate the discounted returns from the stored rewards
    def calculate_returns(self):
        G = 0
        returns = []
        for reward in reversed(self.rewards):
            G = G * self.discount_factor + reward  # Discounted return calculation
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float)

    # Compute the actor-critic loss
    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        returns = self.calculate_returns()

    
        # Stack the padded tensors
        pis = torch.stack(self.pis, dim=0)
        probs = F.softmax(pis, dim=-1)
        dist = Categorical(probs)

        # Log probabilities of the actions taken
        new_log_probs = dist.log_prob(actions)
        


        
        # Calculate advantages
        if self.critic is not None:
            values = torch.stack([self.critic(state) for state in states], dim=0).squeeze()
            advantages = returns - values.detach()  # Using detach to avoid gradient flow
        else:
            advantages = returns
        
        if learning_config['learning_algorithm']=="ppo":
            # PPO objective
            
            # get log probs for the first time (no difference between old and new here)
            for i, action in enumerate(actions):
                if self.old_log_probs[action.item()] is None:
                    self.old_log_probs[action.item()] = Categorical(probs[i]).log_prob(action).item()
                    
            # collect the correspanding old log probs
            old_log_probs = []
            for action in actions:
                old_log_probs.append(self.old_log_probs[action.item()])
            old_log_probs =torch.tensor(old_log_probs)
            
            ratio = torch.exp(new_log_probs - old_log_probs)  # Importance ratio
            
            # update log probs
            for i, action in enumerate(actions):
                self.old_log_probs[action.item()] = Categorical(probs[i]).log_prob(action).item()
            
            p1 = ratio * advantages
            p2 = torch.clamp(ratio, 1 - learning_config['ppo_epsilon'], 1 + learning_config['ppo_epsilon']) * advantages

            actor_loss = -torch.min(p1, p2).mean()  # Minimize the clipped objective
        else:
            actor_loss = -torch.sum(new_log_probs*advantages)
            
        critic_loss = 0
        if self.critic:
            critic_loss = F.mse_loss(values, returns)


        return actor_loss + critic_loss
