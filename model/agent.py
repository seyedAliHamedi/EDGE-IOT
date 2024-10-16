import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from env.utils import *
from model.actor_critic import ActorCritic


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, global_optimizer, barrier,env):
        super(Agent, self).__init__()
        
        self.name = name  # worker/agent name
        self.env = env
        self.global_actor_critic = global_actor_critic  # global shared actor-critic model
        self.global_optimizer = global_optimizer  # shared Adam optimizer
        
        # the local actor-critic and the core scheduler
        self.local_actor_critic = ActorCritic(self.env.devices) # local actor-critic model
       

        self.assigned_job = None  # current job assigned to the agent
        self.runner_flag = mp.Value('b', True)  # flag to control agent's execution loop
        self.barrier = barrier  # barrier for synchronization across processes

    

    def run(self):
        """Main loop where the agent keeps running, processing jobs."""
        while self.runner_flag:
            self.barrier.wait()  # wait for all agents to be synchronized
            if self.assigned_job is None:
                with self.env.lock:
                    self.assigned_job = self.env.assign_job_to_agent()
                if self.assigned_job is None:
                    continue
            # retrive the agent task_queue
            task_queue = self.env.get_agent_queue(self.assigned_job)
            if task_queue is None:
                continue
            
            
            self.time_job = 0
            self.energy_job =0
            self.reward_job = 0
            self.loss_job = 0
            self.fail_job = np.array([0, 0, 0, 0])
            self.usage_job = np.array([0, 0, 0])
            self.path_job = []
            
            utilization = [sum(usage) for usage in self.env.device_usuages]
            gin = gini_coefficient(utilization)
            used_devices_count = sum(1 for usage in self.env.device_usuages if 1 in usage)
            diversity = used_devices_count / len(self.env.devices)
            utilization = torch.tensor(utilization, dtype=torch.float)
            for task in task_queue:
                self.schedule(task,gin,diversity,utilization)
                
            loss_job = self.update()
            self.env.update_monitor(self.time_job, self.energy_job, self.reward_job, loss_job, self.fail_job, self.usage_job, len(task_queue),self.path_job, [sum(usage) for usage in self.env.device_usuages])
            self.assigned_job = None
            

    def stop(self):
        self.runner_flag = False

    def schedule(self, current_task_id,gin,diversity,utilization):
        current_task = self.env.tasks[current_task_id]
        input_state = get_input(current_task, diversity, gin)
        action, path, devices = self.local_actor_critic.choose_action(input_state)
        selected_device_index = action
        if devices:
            selected_device_index = self.env.devices.index(devices[selected_device_index])
        selected_device = self.env.devices[selected_device_index]
        current_task['live_state']["chosen_device_type"] = selected_device["type"]
        current_task_children = [self.env.tasks[child] for child in current_task["successors"]]
        for child in current_task_children:
            child[f"{selected_device['type']}_predecessors"] += 1
        core_index = 0
        (freq, vol) = selected_device['voltages_frequencies'][core_index][0]
        
        utilization = utilization / torch.sum(utilization)
        selected_device_util = utilization[selected_device_index]
        reward, t, e, batteryFail, taskFail, safeFail = self.env.execute_action(pe_ID=selected_device_index,
                                                                                core_i=core_index,
                                                                                freq=freq, volt=vol,
                                                                                task_ID=current_task_id,
                                                                                utilization=selected_device_util, diversity=diversity, gin=gin)
      
        self.local_actor_critic.archive(input_state, action, reward)

        self.reward_job += reward
        self.time_job += t
        self.energy_job += e
        fails = np.array([taskFail + safeFail + batteryFail, batteryFail, taskFail, safeFail])
        self.fail_job += fails
        if selected_device['type'] == 'iot':
            self.usage_job[0] += 1
        if selected_device['type'] == 'mec':
            self.usage_job[1] += 1
        if selected_device['type'] == 'cloud':
            self.usage_job[2] += 1
        self.path_job.append(path)
        


    def update(self):
        self.local_actor_critic.update_regressor()
        
        """Update the global actor-critic based on the local model."""
        loss = self.local_actor_critic.calc_loss()  # compute the loss
        self.global_optimizer.zero_grad()  # zero gradients
        loss.backward()
        # Synchronize local and global models
        for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad

        self.global_optimizer.step()  # update global model
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())  # update local model
        self.local_actor_critic.reset_memory()
        return loss.item()



    def save_models(self):
        print('... saving models ...')
        checkpoint = {
            'model_state_dict': self.global_actor_critic.state_dict(),
        }
        os.makedirs(os.path.dirname(self.global_actor_critic.checkpoint_file), exist_ok=True)
        torch.save(checkpoint, self.global_actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        checkpoint = torch.load(self.global_actor_critic.checkpoint_file)
        self.global_actor_critic.load_state_dict(checkpoint['model_state_dict'])