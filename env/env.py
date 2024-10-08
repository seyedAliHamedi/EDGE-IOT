import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from data.db import Database
from model.actor_critic import ActorCritic
from env.monitor import Monitor
from env.utils import *


class Environment():
    def __init__(self):
        Database().reset()
        print("Database cleared")
        self.devices = Database().get_all_devices()
        self.jobs = Database().get_all_jobs()
        self.tasks = Database().get_all_tasks()
        print("Database initialized")
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.005)
        print("Model initalized")
        self.monitor = Monitor()

    ##### Functionality

    def execute_action(self, pe_ID, core_i, freq, volt, task_ID):
        pe = Database().get_device(pe_ID)
        task = Database().get_task(task_ID)

        fail_flags = [0, 0, 0]
        if task["is_safe"] and not pe['handleSafeTask']:
            # fail : assigned safe task to unsafe device
            fail_flags[0] = 1
        if task["task_kind"] not in pe["acceptableTasks"]:
            # fail : assigned a kind of task to the inappropriate device
            fail_flags[1] = 1

        total_t, total_e = calc_total(pe, task, core_i, 0)
        reg_e = total_e
        reg_t = total_t

        if sum(fail_flags) > 0:
            return sum(fail_flags) * reward_function(punish=True), 0, 0, 0, fail_flags[1], fail_flags[0]

        battery_drain_punish = batteryFail = 0
        if learning_config["drain_battery"]:
            battery_drain_punish, fail_flags[2] = checkBatteryDrain(reg_e,pe)
        if fail_flags[2]:
            return reward_function(punish=True), 0, 0, 1, 0, 0
        if learning_config['regularize_output']:
            reg_t = regularize_output(total_t=total_t)
            reg_e = regularize_output(total_e=total_e)
        return reward_function(t=reg_t, e=reg_e) + battery_drain_punish, reg_t, reg_e, fail_flags[2], fail_flags[1], \
            fail_flags[0]

    def add_device(self):
        # Add a new random device using the Database method
        new_device_id = len(self.devices)  # You can use a more complex ID generation logic if needed
        device = Database().add_device(new_device_id)

        # Refresh the devices list after adding a device
        self.devices = Database().get_all_devices()
        self.actor_critic.add_new_device(device)

    def remove_device(self):
        # Randomly remove a device
        if len(self.devices) > 1:
            device_index = np.random.randint(0, len(self.devices) - 1)
            device_id = self.devices[device_index]['id']
            if self.devices[device_index]['type'] == 'cloud':
                return self.remove_device()

            # Remove the selected device from the Database
            Database().remove_device(device_id)

            # Refresh the devices list after removing the device
            self.devices = Database().get_all_devices()
            self.actor_critic.remove_new_device(device_index)

    def run(self):
        print("Starting...")
        starting_time = time.time()
        for job_id in range(len(self.jobs)):

            if job_id == 10000:
                self.monitor.plot_histories()
                print("scaliblity ---------")

            # Dynamically add/remove devices
            if learning_config['scalability'] and job_id > 10000:
                if np.random.random() < learning_config['add_device_iterations']:
                    print("device Add")
                    self.add_device()
                if np.random.random() < learning_config['remove_device_iterations']:
                    print("device Removed")
                    self.remove_device()

            if ((job_id / len(self.jobs)) * 100) % 10 == 0:
                print(f"{((job_id / len(self.jobs)) * 100)}% done in {int(time.time() - starting_time)} seconds")

            time_job = energy_job = reward_job = loss_job = 0
            fail_job = np.array([0, 0, 0, 0])
            usage_job = np.array([0, 0, 0])
            path_job = []

            tasks = Database().get_job(job_id)["tasks_ID"]
            for task_id in tasks:
                current_task = Database().get_task_norm(task_id)
                input_state = get_input(current_task)

                action, path, devices = self.actor_critic.choose_action(input_state)
                selected_device_index = action
                if devices:
                    selected_device_index = self.devices.index(devices[selected_device_index])

                selected_device = self.devices[selected_device_index]
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
                # Database().update_core_occupied()
                self.devices=Database().get_all_devices()

            loss_job = self.actor_critic.calc_loss()
            self.monitor.update(time_job, energy_job, reward_job, loss_job, fail_job, usage_job, len(tasks), path_job)

            self.optimizer.zero_grad()
            loss_job.backward()
            self.optimizer.step()

            if job_id % 10 == 0:
                self.actor_critic.update_regressor()

            self.actor_critic.reset_memory()

        self.monitor.save_results()
        self.monitor.plot_histories()
        print("----------- COMPLETED------------")
