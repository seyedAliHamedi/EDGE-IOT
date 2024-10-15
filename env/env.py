from collections import deque
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import multiprocessing as mp

from data.gen import Generator
from model.actor_critic import ActorCritic
from env.monitor import Monitor
from env.utils import *
from model.agent import Agent
from model.shared_adam import SharedAdam

class Environment:
    def __init__(self,barrier):
        manager=mp.Manager()
        self.initialize(manager=manager)
        self.util = Utility(self.devices)
        self.monitor = Monitor()
        self.device_usuages = manager.list([deque([1],maxlen=100) for i in range(300)])
        self.lock = mp.Lock()
        self.barrier=barrier

    def initialize(self,manager):
        print("initialize Envionment")
        self.devices = manager.list(Generator.get_devices())
        self.jobs = manager.list(Generator.get_jobs())
        self.remaining_jobs = manager.list(self.jobs)
        self.tasks = manager.list(Generator.get_tasks())
        print("Data loaded")


    def run(self):
        print("Starting...")
        iteration = 0
        while iteration <= 1000:
            self.barrier.wait()
            print("RUNNING ",iteration)
            self.monitor.run(iteration)

            # if job_id > 20000:
                # self.change_env()
            
            self.clean_dead_iot()

            iteration +=1

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
        with self.lock:
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
                device_index = np.random.randint(0, len(self.devices) - 1)
                if self.devices[device_index]['type'] == 'cloud':
                    return self.remove_device()
            else:
                print("iot battery dead removed")
            # Remove the selected device from the Database
            del self.devices[device_index]
            # Refresh the devices list after removing the device
            self.actor_critic.remove_new_device(device_index)
    
    def clean_dead_iot(self):
        for device_index,device in enumerate(self.devices):
            if device['type']=='iot' and device['live_state']['battery_now'] < device['ISL']*100:
                self.remove_device(device_index=device_index)
                
    ######### MULTI AGENT
    def assign_job_to_agent(self):
        job = self.remaining_jobs[0]
        self.remaining_jobs[:] = self.remaining_jobs[1:]
        return self.jobs.index(job)
    
    
    def get_agent_queue(self,job_id):
        return self.jobs[job_id]["tasks_ID"]