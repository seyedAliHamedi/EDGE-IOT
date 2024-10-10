from collections import deque
import os
import time
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
        self.device_usuages = [deque(maxlen=1000) for i in range(len(self.devices))]

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

            utilization = None
            if job_id > 10000:
                self.change_env()
                utilization = [sum(usage) for usage in self.device_usuages]

            time_job = energy_job = reward_job = loss_job = 0
            fail_job = np.array([0, 0, 0, 0])
            usage_job = np.array([0, 0, 0])
            path_job = []

            task_list = self.jobs[job_id]["tasks_ID"]
            for task_id in task_list:
                current_task = self.tasks[task_id]
                input_state = self.util.get_input(current_task)

                action, path, devices = self.actor_critic.choose_action(input_state, utilization)

                selected_device_index = action
                if devices:
                    selected_device_index = self.devices.index(devices[selected_device_index])

                selected_device = self.devices[selected_device_index]
                current_task["chosen_device_type"] = selected_device["type"]
                current_task_children = [self.tasks[child] for child in current_task["successors"]]
                for child in current_task_children:
                    child[f"{selected_device['type']}_predecessors"] += 1

                core_index = -1
                # for i,core in enumerate(selected_device['occupied_cores']):
                #     if core==-1:
                #         core_index = i
                #         Database().set_core_occupied(selected_device["id"], core_index)
                #         break
                # if core_index==-1:
                #     #TODO device full
                #     print("+_+_+_+_+_+_+ DEVICE FULL __+_+_+_+_+_+_+_")
                (freq, vol) = selected_device['voltages_frequencies'][core_index][0]
                reward, t, e, batteryFail, taskFail, safeFail = self.execute_action(pe_ID=selected_device_index,
                                                                                    core_i=core_index,
                                                                                    freq=freq, volt=vol,
                                                                                    task_ID=task_id)
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

            if job_id % 10 == 0:
                self.actor_critic.update_regressor()

            # if job_id % 1000 ==0:
            #     self.save_models()
            #     self.load_models()

            self.actor_critic.reset_memory()

    def change_env(self):
        if learning_config['scalability']:
            if np.random.random() < learning_config['add_device_iterations']:
                print("device Add")
                self.add_device()
            if np.random.random() < learning_config['remove_device_iterations']:
                print("device Removed")
                self.remove_device()

    ##### Functionality

    def execute_action(self, pe_ID, core_i, freq, volt, task_ID):
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

        battery_drain_punish = batteryFail = 0
        if learning_config["drain_battery"]:
            battery_drain_punish, fail_flags[2] = self.util.checkBatteryDrain(reg_e, pe)
        if fail_flags[2]:
            return reward_function(punish=True), 0, 0, 1, 0, 0
        if learning_config['regularize_output']:
            reg_t = self.util.regularize_output(total_t=total_t)
            reg_e = self.util.regularize_output(total_e=total_e)
        return reward_function(t=reg_t, e=reg_e) + battery_drain_punish, reg_t, reg_e, fail_flags[2], fail_flags[1], \
            fail_flags[0]

    def add_device(self):
        # Add a new random device using the Database method
        device = Generator.generate_random_device()
        self.devices.append(device)
        # Refresh the devices list after adding a device
        self.actor_critic.add_new_device(device)

    def remove_device(self):
        # Randomly remove a device
        if len(self.devices) > 1:
            device_index = np.random.randint(0, len(self.devices) - 1)
            if self.devices[device_index]['type'] == 'cloud':
                return self.remove_device()
            # Remove the selected device from the Database

            del self.devices[device_index]

            # Refresh the devices list after removing the device
            self.actor_critic.remove_new_device(device_index)
