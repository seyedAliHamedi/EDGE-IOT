import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data.config import learning_config
from data.db import Database

class Monitor():
    def __init__(self):
        self.config = learning_config
        self.avg_time_history = np.array([])
        self.avg_energy_history = np.array([])
        self.avg_fail_history = np.array([0,0,0])
        self.avg_loss_history = np.array([])
        self.avg_reward_history = np.array([])

        self.iot_usage = []
        self.mec_usage = []
        self.cc_usage = []
        self.path_history = []
        

    def save_results(self):
        num_epoch = len(Database().get_all_jobs())
        half_num_epoch = num_epoch//2

        new_epoch_data = {
            "Setup": learning_config['rewardSetup'],
            "Punishment": learning_config['punish'],

            "Average Loss":  sum(self.avg_loss_history)/num_epoch,
            "Last Epoch Loss": self.avg_loss_history[-1],

            "Task Converge": int(np.argmax(np.flip(self.avg_fail_history[:, 1]) != 0)),
            "Task Fail Percentage": np.count_nonzero(self.avg_fail_history[:, 1])/len(self.avg_fail_history[:, 1]),
            "Safe Converge": int(np.argmax(np.flip(self.avg_fail_history[:, 2]) != 0)),
            "Safe Fail Percentage": np.count_nonzero(self.avg_fail_history[:, 2])/len(self.avg_fail_history[:, 2]),

            "Average Time": sum(self.avg_time_history)/num_epoch,
            "Last Epoch Time": self.avg_time_history[-1],

            "Average Energy": sum(self.avg_energy_history)/num_epoch,
            "Last Epoch Energy":  self.avg_energy_history[-1],

            "Average Reward":  sum(self.avg_reward_history)/num_epoch,
            "Last Epoch Reward": self.avg_reward_history[-1],

            "First 10 Avg Time": np.mean(self.avg_time_history[:10]),
            "Mid 10 Avg Time": np.mean(self.avg_time_history[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Time": np.mean(self.avg_time_history[:-10]),

            "First 10 Avg Energy": np.mean(self.avg_energy_history[:10]),
            "Mid 10 Avg Energy": np.mean(self.avg_energy_history[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Energy": np.mean(self.avg_energy_history[:-10]),

            "First 10 Avg Reward": np.mean(self.avg_reward_history[:10]),
            "Mid 10 Avg Reward": np.mean(self.avg_reward_history[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Reward": np.mean(self.avg_reward_history[:-10]),


            "First 10 Avg Loss": np.mean(self.avg_loss_history[:10]),
            "Mid 10 Avg Loss": np.mean(self.avg_loss_history[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Loss": np.mean(self.avg_loss_history[:-10]),

            "First 10 (total, task, safe) Fail": str(np.mean(self.avg_fail_history[:10], axis=0)),
            "Mid 10 (total, task, safe) Fail":  str(np.mean(self.avg_fail_history[half_num_epoch:half_num_epoch + 10], axis=0)),
            "Last 10 (total, task, safe) Fail": str(np.mean(self.avg_fail_history[:-10], axis=0)),
        }
        new_epoch_data_list = [new_epoch_data]

        df = None
        if os.path.exists(learning_config['result_summery_path']):
            df = pd.read_csv(learning_config['result_summery_path'])
            new_df = pd.DataFrame(new_epoch_data_list)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(new_epoch_data_list)

        df.to_csv(learning_config['result_summery_path'], index=False)


    def plot_histories(self,punish=0, epsilon=0, init_explore_rate=0, explore_rate=0, exp_counter=0):
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))

        plt.suptitle(
            f"Training History with setup {learning_config['rewardSetup']}, initial punish: {learning_config['init_punish']}, final punish: {punish}", fontsize=16, fontweight='bold')

        loss_values = self.avg_loss_history
        axs[0, 0].plot(loss_values, label='Average Loss',
                    color='blue', marker='o')  # Add markers for clarity
        axs[0, 0].set_title('Average Loss History')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot for average time history
        time_values = np.array(self.avg_time_history)  # Ensure data is in numpy array
        axs[0, 1].plot(time_values, label='Average Time', color='red', marker='o')
        axs[0, 1].set_title('Average Time History')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Time')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        time_lower_bound = 0.00625
        time_middle_bound = 0.0267
        time_upper_bound = 1
        # axs[0, 1].axhline(y=time_lower_bound, color='blue',
        #                   linestyle='--', label='Lower Bound (0.00625)')
        # axs[0, 1].axhline(y=time_middle_bound, color='green',
        #                   linestyle='--', label='Middle Bound (0.0267)')
        # axs[0, 1].axhline(y=time_upper_bound, color='red',
        #                   linestyle='--', label='Upper Bound (1)')
        axs[0, 1].legend()

        # Plot for average energy history
        energy_values = np.array(self.avg_energy_history)
        axs[1, 0].plot(energy_values, label='Average Energy',
                    color='green', marker='o')
        axs[1, 0].set_title('Average Energy History')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Energy')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        energy_lower_bound = 0.0000405
        energy_middle_bound = 0.100746
        energy_upper_bound = 1.2
        # axs[1, 0].axhline(y=energy_lower_bound, color='blue',
        #                   linestyle='--', label='Lower Bound (0.0000405)')
        # axs[1, 0].axhline(y=energy_middle_bound, color='green',
        #                   linestyle='--', label='Middle Bound (0.100746)')
        # axs[1, 0].axhline(y=energy_upper_bound, color='red',
        #                   linestyle='--', label='Upper Bound (1.2)')
        axs[1, 0].legend()

        # Plot for average fail history
        fail_values = np.array(self.avg_fail_history)
        axs[1, 1].plot(fail_values, label='Average Fail',
                    color='purple', marker='o')
        axs[1, 1].set_title('Average Fail History')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Fail Count')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Plot for devices usage history
        axs[2, 0].plot(self.iot_usage, label='IoT Usage', color='blue', marker='o')
        axs[2, 0].plot(self.mec_usage, label='MEC Usage', color='orange', marker='x')
        axs[2, 0].plot(self.cc_usage, label='Cloud Usage', color='green', marker='s')
        axs[2, 0].set_title('Devices Usage History')
        axs[2, 0].set_xlabel('Epochs')
        axs[2, 0].set_ylabel('Usage')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Heatmap for path history
        output_classes = ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
        path_counts = np.zeros((len(self.path_history), len(output_classes)))

        for epoch in range(len(self.path_history)):
            epoch_paths = self.path_history[epoch]

            for path in epoch_paths:
                path_index = output_classes.index(path)
                path_counts[epoch, path_index] += 1
        sns.heatmap(path_counts, cmap="YlGnBu",
                    xticklabels=output_classes, ax=axs[2, 1])
        axs[2, 1].set_title(
            f'Path History Heatmap - All Epochs\n(r: {learning_config['rewardSetup']}, p: {learning_config['init_punish']}, ep: {epsilon}, exp_rate: {init_explore_rate:.5f} - {explore_rate:.5f}, exp_times: {exp_counter})')
        axs[2, 1].set_xlabel('Output Classes')
        axs[2, 1].set_ylabel('Epochs')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(learning_config['result_plot_path'])



    def update(self,time_epoch,energy_epoch,reward_epoch,loss_epoch,fail_epoch,usage_epoch,num_episodes,path_job):
        
        avg_time = time_epoch / num_episodes
        avg_energy = energy_epoch / num_episodes
        avg_reward = reward_epoch / num_episodes
        avg_loss = loss_epoch/num_episodes
        avg_fail = [elem/num_episodes for elem in fail_epoch]

        avg_loss = avg_loss.detach().numpy()
        self.avg_loss_history = np.append(self.avg_loss_history,avg_loss)
        self.avg_reward_history = np.append(self.avg_reward_history,avg_reward)
        self.avg_time_history = np.append(self.avg_time_history,avg_time)
        self.avg_energy_history = np.append(self.avg_energy_history,avg_energy)
        self.avg_fail_history = np.vstack([self.avg_fail_history,avg_fail])
        
        self.iot_usage.append(usage_epoch[0])
        self.mec_usage.append(usage_epoch[1])
        self.cc_usage.append(usage_epoch[2])
        self.path_history.append(path_job)