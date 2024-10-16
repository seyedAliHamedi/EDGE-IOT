from collections import deque
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from data.gen import Generator
from model.actor_critic import ActorCritic
from env.monitor import Monitor
from env.utils import *


class Environment():
    def __init__(self):
        self.initialize()
        self.util = Utility(self.devices)
        self.actor_critic = ActorCritic(devices=self.devices)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.005)
        self.monitor = Monitor()
        self.device_usuages = [deque([1],maxlen=100) for i in range(len(self.devices))]
        self.stablized =False

    def initialize(self):
        print("initialize Envionment")
        self.devices = Generator.get_devices()
        self.jobs = Generator.get_jobs()
        self.tasks = Generator.get_tasks()
        print("Data loaded")

    def save_models(self):
        print('... saving models ...')
        checkpoint = {
            'model_state_dict': self.actor_critic.state_dict(),
        }
        os.makedirs(os.path.dirname(self.actor_critic.checkpoint_file), exist_ok=True)
        torch.save(checkpoint, self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        checkpoint = torch.load(self.actor_critic.checkpoint_file)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])

    def run(self):
        print("Starting...")
        for job_id in range(learning_config['num_epoch'] - 1):
            self.monitor.run(job_id)
            
            if self.stablized:
                self.change_env()

            utilization = [sum(usage) for usage in self.device_usuages]
            diversity = None
            gin = self.util.gini_coefficient(utilization)
            used_devices_count = sum(1 for usage in self.device_usuages if 1 in usage)
            diversity = used_devices_count / len(self.devices)
            utilization = torch.tensor(utilization, dtype=torch.float)
            
            
                
            if job_id > 20000 and all(x <= 0.15 for x in self.monitor.avg_fail_history[:,0][-1000:]) and not self.stablized:
                print("STABLIZED")
                self.stablized =True
            
            self.clean_dead_iot()

            time_job = energy_job = reward_job = loss_job = 0
            fail_job = np.array([0, 0, 0, 0])
            usage_job = np.array([0, 0, 0])
            path_job = []

            task_list = self.jobs[job_id]["tasks_ID"]
            for task_id in task_list:
                current_task = self.tasks[task_id]
                input_state = self.util.get_input(current_task, diversity, gin)
                action, path, devices = self.actor_critic.choose_action(input_state)

                selected_device_index = action
                if devices:
                    selected_device_index = self.devices.index(devices[selected_device_index])

                selected_device = self.devices[selected_device_index]
                current_task['live_state']["chosen_device_type"] = selected_device["type"]
                current_task_children = [self.tasks[child] for child in current_task["successors"]]
                for child in current_task_children:
                    child[f"{selected_device['type']}_predecessors"] += 1

                core_index = 0
                (freq, vol) = selected_device['voltages_frequencies'][core_index][0]
                
                utilization = utilization / torch.sum(utilization)
                selected_device_util = utilization[selected_device_index]
                reward, t, e, batteryFail, taskFail, safeFail = self.execute_action(pe_ID=selected_device_index,
                                                                                    core_i=core_index,
                                                                                    freq=freq, volt=vol,
                                                                                    task_ID=task_id,
                                                                                    utilization=selected_device_util, diversity=diversity, gin=gin)
                for i, _ in enumerate(self.device_usuages):
                    if i == selected_device_index:
                        self.device_usuages[i].append(1)
                    else:
                        self.device_usuages[i].append(0)

                self.actor_critic.archive(input_state, action, reward)

                reward_job += reward
                time_job += t
                energy_job += e
                fails = np.array([taskFail + safeFail + batteryFail, batteryFail, taskFail, safeFail])
                fail_job += fails
                if selected_device['type'] == 'iot':
                    usage_job[0] += 1
                if selected_device['type'] == 'mec':
                    usage_job[1] += 1
                if selected_device['type'] == 'cloud':
                    usage_job[2] += 1
                path_job.append(path)
            
            loss_job = self.actor_critic.calc_loss()
            self.monitor.update(time_job, energy_job, reward_job, loss_job, fail_job, usage_job, len(task_list),
                                path_job, [sum(usage) for usage in self.device_usuages])

            self.optimizer.zero_grad()
            loss_job.backward()
            self.optimizer.step()

            self.actor_critic.update_regressor()


            self.actor_critic.reset_memory()

    def change_env(self):
        if learning_config['scalability']:
            if float("{:.5f}".format(np.random.random())) < learning_config['add_device_iterations']:
                print("device Add")
                self.add_device()
            if float("{:.5f}".format(np.random.random())) < learning_config['remove_device_iterations']:
                print("device Removed")
                self.remove_device()

    ##### Functionality

    def execute_action(self, pe_ID, core_i, freq, volt, task_ID, utilization, diversity, gin):
        pe = self.devices[pe_ID]
        task = self.tasks[task_ID]
        task_pres = [self.tasks[pre_id] for pre_id in task["predecessors"]]

        fail_flags = [0, 0, 0]
        if task["is_safe"] and not pe['is_safe']:
            # fail : assigned safe task to unsafe device
            fail_flags[0] = 1
        if task["task_kind"] not in pe["acceptable_tasks"]:
            # fail : assigned a kind of task to the inappropriate device
            fail_flags[1] = 1

        total_t, total_e = calc_total(pe, task, task_pres, core_i, 0)
        reg_e = total_e
        reg_t = total_t

        if sum(fail_flags) > 0:
            return sum(fail_flags) * reward_function(punish=True), 0, 0, 0, fail_flags[1], fail_flags[0]

        battery_drain_punish = 0
        if learning_config["drain_battery"]:
            battery_drain_punish, fail_flags[2] = self.util.checkBatteryDrain(reg_e, pe)
            
            if fail_flags[2]:
                return reward_function(punish=True), 0, 0, 1, 0, 0
        
        lambda_penalty = 0
        if learning_config['utilization']            :
            lambda_diversity = learning_config["max_lambda"] * (1 - diversity)
            lambda_gini = learning_config["max_lambda"] * gin
            lambda_penalty = learning_config["alpha_diversity"] * lambda_diversity + learning_config["alpha_gin"] * lambda_gini

        if learning_config['regularize_output']:
            reg_t = self.util.regularize_output(total_t=total_t)
            reg_e = self.util.regularize_output(total_e=total_e)
        return reward_function(t=reg_t, e=reg_e) * (1 - lambda_penalty * utilization) + battery_drain_punish, reg_t, reg_e, fail_flags[2], fail_flags[1], fail_flags[0]
    
    def add_device(self):
        # Add a new random device using the Database method
        device = Generator.generate_random_device()
        self.devices.append(device)
        self.device_usuages.append(deque([1],maxlen=100))
        # Refresh the devices list after adding a device
        self.actor_critic.add_new_device(device)

    def remove_device(self,device_index=None):
        # Randomly remove a device
        if len(self.devices) > 1:
            if device_index is None:
                device_index = random.choices(range(len(self.devices)), weights=[1/(x+1) for x in [sum(usage) for usage in self.device_usuages]], k=1)[0]
                if self.devices[device_index]['type'] == 'cloud':
                    return self.remove_device()
            else:
                print("iot battery dead removed")
            # Remove the selected device from the Database
            del self.devices[device_index]
            del self.device_usuages[device_index]
            # Refresh the devices list after removing the device
            self.actor_critic.remove_new_device(device_index)
    def clean_dead_iot(self):
        for device_index,device in enumerate(self.devices):
            if device['type']=='iot' and device['live_state']['battery_now'] < device['ISL']*100:
                self.remove_device(device_index=device_index)