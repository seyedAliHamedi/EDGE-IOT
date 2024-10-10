import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from config import learning_config
from model.trees.ClusTree import ClusTree
from model.utils import get_tree, get_critic
from env.utils import *
from sklearn.metrics import  r2_score


class ActorCritic(nn.Module):
    def __init__(self,devices):
        super(ActorCritic, self).__init__()
        self.discount_factor = learning_config['discount_factor']
        self.devices=devices
        self.actor = get_tree(self.devices)  # Initialize actor
        self.critic = get_critic()  # Critic could be None
        self.checkpoint_file = learning_config['checkpoint_file_path']
        self.reset_memory()
        self.old_log_probs = {i: None for i in range(len(self.devices))} 
        self.utilization_factor = 0.5 

    def add_new_device(self,new_device):
        self.actor.add_device(new_device) 
        self.old_log_probs[len(self.devices)-1]=None
        
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
        if isinstance(self.actor, ClusTree)   :
            p, path,devices = self.actor(x)  # Get policy distribution and path from the actor
        else:
            p, path = self.actor(x)  # Get policy distribution and path from the actor
            devices = None  # tree does not return devices
            
        v = self.critic(x) if self.critic is not None else None  # Value estimate from the critic, if applicable
        return p, path, devices, v

    # Select action based on actor's policy
    def choose_action(self, observation,utilization=None):
        state = torch.tensor(observation, dtype=torch.float)
        pi, path, devices, _ = self.forward(state)
        
        self.pis.append(pi)  # Store policy distribution

        if utilization is not None and learning_config['utilization']:
            utilization = torch.tensor(utilization, dtype=torch.float)  # Ensure utilization is a tensor
        
            # Normalize utilization to [0, 1]
            utilization = utilization / torch.max(utilization)  # Scale to max utilization value

            # Dynamic temperature adjustment based on utilization
            # Higher utilization leads to a higher temperature
            
            temperature = 1.0 + self.utilization_factor * utilization
            if self.utilization_factor >= 0.5 and self.utilization_factor <=1:
                self.utilization_factor += learning_config['utilization_eps']
            if self.utilization_factor == 1:
                learning_config['utilization_eps'] = -1*learning_config['utilization_eps']
            
            # Use a more pronounced adjustment
            pi_adjusted = pi / temperature
            
            # Normalize adjusted policy distribution
            probs = F.softmax(pi_adjusted, dim=-1)

        else:
            probs = F.softmax(pi, dim=-1)

            
        
        dist = Categorical(probs)  # Create a categorical distribution over actions
        action = dist.sample()
        
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
        # max_length = max(p.size(0) for p in self.pis)  # Find the maximum length
        # padded_pis = [F.pad(p, (0, max_length - p.size(0)), 'constant', 0) for p in self.pis]

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
    
    def update_regressor(self):
        def update_leaf_nodes(node):
            if node.depth == node.max_depth:
                devices =  node.devices if isinstance(self.actor, ClusTree) else self.devices
                dist = node.prob_dist
                pe_data = torch.tensor(
                    [extract_pe_data(device) for device in devices],
                    dtype=torch.float32
                )
                pred=  node.logit_regressor(pe_data)
                loss = F.mse_loss(pred.squeeze(),dist)
                node.logit_optimizer.zero_grad()
                loss.backward()
                node.logit_optimizer.step()
                return
            
            if node.left:
                update_leaf_nodes(node.left)
            if node.right:
                update_leaf_nodes(node.right)

        # Start from the root actor
        actor = self.actor
        update_leaf_nodes(actor)

