import numpy as np
from collections import deque

import multiprocessing as mp

from env.utils import *
from data.gen import Generator
from env.monitor import Monitor

class Environment:
    def __init__(self,barrier):
        self.barrier=barrier
        manager = mp.Manager()
        self.initialize(manager=manager)
        self.util = Utility(self.devices)
        self.monitor = Monitor(manager=manager)
        self.device_usuages = manager.list([manager.list([1]) for i in range(len(self.devices))])
        self.lock = mp.Lock()

    def initialize(self,manager):
        print("initialize Envionment")
        self.devices = manager.list(Generator.get_devices())
        self.jobs = manager.list(Generator.get_jobs())
        self.tasks = manager.list(Generator.get_tasks())
        print("Data loaded")
        self.remaining_jobs = manager.list(self.jobs)


    def run(self):
        print("Environment Starting ")
        iteration = 0
        while iteration <= learning_config['num_iteration']:
            self.barrier.wait()
            print(iteration)
            self.monitor.run(iteration)

            # if job_id > 20000:
                # self.change_env(manager)
            
            self.clean_dead_iot()
            iteration +=1



    def execute_action(self, pe_ID, core_i, freq, volt, task_ID, utilization, diversity, gin):
        try:
            pe = self.devices[pe_ID]
            task = self.tasks[task_ID]
            task_pres = [self.tasks[pre_id] for pre_id in task["predecessors"]]
            device_usuages = self.device_usuages
        except Exception as e:
            print("Retrying execute action")
            return self.execute_action( pe_ID, core_i, freq, volt, task_ID, utilization, diversity, gin)
        
        
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
            battery_drain_punish, fail_flags[2] = checkBatteryDrain(reg_e, pe)

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
            
        for i, _ in enumerate(device_usuages):
          if i == pe_ID:
              device_usuages[i].append(1)
          else:
              device_usuages[i].append(0)
        
        return reward_function(t=reg_t, e=reg_e) * (1 - lambda_penalty * utilization) + battery_drain_punish, reg_t, reg_e, fail_flags[2], fail_flags[1], fail_flags[0]
    def update_monitor(self, time_epoch, energy_epoch, reward_epoch, loss_epoch, fail_epoch, usage_epoch, num_episodes, path_job, device_usage):
        try:
            self.monitor.update(time_epoch, energy_epoch, reward_epoch, loss_epoch, fail_epoch, usage_epoch, num_episodes, path_job, device_usage)
        except:
            print("Retrying update monitor")
            return self.update_monitor(time_epoch, energy_epoch, reward_epoch, loss_epoch, fail_epoch, usage_epoch, num_episodes, path_job, device_usage)
    ##### Functionality
    def change_env(self,manager):
        if learning_config['scalability']:
            if float("{:.5f}".format(np.random.random())) < learning_config['add_device_iterations']:
                print("device Add")
                self.add_device(manager)
            if float("{:.5f}".format(np.random.random())) < learning_config['remove_device_iterations']:
                print("device Removed")
                self.remove_device()


    def add_device(self,manager):
        # Add a new random device using the Database method
        device = Generator.generate_random_device()
        self.devices.append(device)
        self.device_usuages.append(manager.list())
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
        return job['id']
    
    def get_agent_queue(self,job_id):
        return self.jobs[job_id]["tasks_ID"]