import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.optim as optim

from data.db import Database
from model.actor_critic import ActorCritic
from monitor import Monitor
from utils import *

class Environment():
    def __init__(self):
        self.devices=Database().get_all_devices()
        self.jobs=Database().get_all_jobs()
        self.tasks=Database().get_all_tasks()
        
        
        self.actor_critic =ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(),lr=0.005)
        
        self.monitor = Monitor()
        
        ##### Functionality
    def execute_action(self, pe_ID, core_i, freq, volt, task_ID):
        pe = Database().get_device(pe_ID)
        task = Database().get_task(task_ID)


        fail_flags = [0, 0]
        if task["is_safe"] and not pe['handleSafeTask']:
            # fail : assigned safe task to unsafe device
            fail_flags[0] = 1
        if task["task_kind"] not in pe["acceptableTasks"]:
            # fail : assigned a kind of task to the inappropriate device
            fail_flags[1] = 1

        if sum(fail_flags) > 0:
            return sum(fail_flags) * reward_function(punish=True), 0,0, fail_flags[1], fail_flags[0]

        
        total_t, total_e  = calc_total(pe, task, core_i,0)
        reg_e = total_e
        reg_t = total_t
        # if self.shouldRegular:
            # reg_t = regularize_any(total_t, 1)
            # reg_e = regularize_any(total_e, 2)
        return reward_function(t=reg_t , e=reg_e), reg_t,reg_e, fail_flags[1], fail_flags[0]
    

    def run(self):
        for job_id in range(len(self.jobs)):
            time_job = energy_job = reward_job = loss_job = 0
            fail_job = usage_job = np.array([0,0,0])
            path_job = []

            tasks = Database().get_job(job_id)["tasks_ID"]
            for task_id in tasks:
                current_task = Database().get_task_norm(task_id)
                input_state = get_input(current_task)

                action, path ,devices= self.actor_critic.choose_action(input_state)
                selected_device_index = action
                if devices:
                    selected_device_index = self.devices.index(devices[selected_device_index])
                    
                path_job.append(path)
                
                    
                reward, t, e, taskFail, safeFail = self.execute_action(pe_ID=selected_device_index,core_i=0,freq=1,volt=1,task_ID=task_id)
                
                self.actor_critic.archive(input_state, action, reward)
                
                reward_job += reward
                time_job += t
                energy_job += e
                fails = np.array([taskFail + safeFail, taskFail, safeFail])
                fail_job += fails
                
            loss_job=self.actor_critic.calc_loss()
            self.monitor.update(time_job,energy_job,reward_job,loss_job,fail_job,usage_job,len(tasks),path_job)
            
            self.optimizer.zero_grad()
            loss_job.backward()
            self.optimizer.step()
            loss_job=self.actor_critic.reset_memory()
            
        self.monitor.save_results()
        self.monitor.plot_histories()
        print("COMPLETED")
        
    
   