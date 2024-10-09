import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import learning_config,devices_config


class Monitor():
    def __init__(self):
        self.config = learning_config
        self.avg_time_history = np.array([])
        self.avg_energy_history = np.array([])
        self.avg_fail_history = np.array([0, 0,0, 0])
        self.avg_loss_history = np.array([])
        self.avg_reward_history = np.array([])

        self.iot_usage = []
        self.mec_usage = []
        self.cc_usage = []
        self.device_usage=[]
        self.path_history = []
        self.starting_time = time.time()
        
    def run(self,job_id):
        
        if job_id>0 and ((job_id / learning_config['num_epoch']) * 100) % 10 == 0:
                print(f"{((job_id / learning_config['num_epoch']) * 100)}% done in {int(time.time() - self.starting_time)} seconds")
                self.plot_histories()
        
        if job_id==learning_config['num_epoch']-2:
            self.save_results()
            self.plot_histories()
            print("----------- COMPLETED------------")
        
    def save_results(self):
        num_epoch = learning_config['num_epoch']
        half_num_epoch = num_epoch // 2

        new_epoch_data = {
            "Setup": learning_config['rewardSetup'],
            "Punishment": learning_config['init_punish'],

            "Average Loss": sum(self.avg_loss_history) / num_epoch,
            "Last Epoch Loss": self.avg_loss_history[-1],
            
            "Battery Converge": int(np.argmax(np.flip(self.avg_fail_history[:,1]) != 0)),
            "Battery Fail Percentage": np.count_nonzero(self.avg_fail_history[:, 1]) / len(self.avg_fail_history[:, 1]),
            "Task Converge": int(np.argmax(np.flip(self.avg_fail_history[:, 1]) != 0)),
            "Task Fail Percentage": np.count_nonzero(self.avg_fail_history[:, 2]) / len(self.avg_fail_history[:, 2]),
            "Safe Converge": int(np.argmax(np.flip(self.avg_fail_history[:, 3]) != 0)),
            "Safe Fail Percentage": np.count_nonzero(self.avg_fail_history[:, 3]) / len(self.avg_fail_history[:, 3]),

            "Average Time": sum(self.avg_time_history) / num_epoch,
            "Last Epoch Time": self.avg_time_history[-1],

            "Average Energy": sum(self.avg_energy_history) / num_epoch,
            "Last Epoch Energy": self.avg_energy_history[-1],

            "Average Reward": sum(self.avg_reward_history) / num_epoch,
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
            "Mid 10 (total, task, safe) Fail": str(
                np.mean(self.avg_fail_history[half_num_epoch:half_num_epoch + 10], axis=0)),
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

    def plot_histories(self, punish=0, epsilon=0, init_explore_rate=0, explore_rate=0, exp_counter=0):
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 2)

        plt.suptitle(
            f"Training History with setup {learning_config['rewardSetup']}, initial punish: {learning_config['init_punish']}, final punish: {punish}",
            fontsize=16, fontweight='bold')

        loss_values = self.avg_loss_history
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(loss_values, label='Average Loss', color='blue', marker='o')  # Add markers for clarity
        ax1.set_title('Average Loss History')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot for average time history
        time_values = np.array(self.avg_time_history)  # Ensure data is in numpy array
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_values, label='Average Time', color='red', marker='o')
        ax2.set_title('Average Time History')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Time')
        ax2.legend()
        ax2.grid(True)

        # Plot for average energy history
        energy_values = np.array(self.avg_energy_history)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(energy_values, label='Average Energy', color='green', marker='o')
        ax3.set_title('Average Energy History')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Energy')
        ax3.legend()
        ax3.grid(True)

        # Plot for average fail history
        fail_values = np.array(self.avg_fail_history[:, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(fail_values, label='Average Fail', color='purple', marker='o')
        ax4.set_title('Average Fail History')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Fail Count')
        ax4.legend()
        ax4.grid(True)

        # Plot for devices usage history
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.iot_usage, label='IoT Usage', color='blue', marker='o')
        ax5.plot(self.mec_usage, label='MEC Usage', color='orange', marker='x')
        ax5.plot(self.cc_usage, label='Cloud Usage', color='green', marker='s')
        ax5.set_title('Devices Usage History')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Usage')
        ax5.legend()
        ax5.grid(True)

        # Heatmap for path history
        output_classes = ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
        path_counts = np.zeros((len(self.path_history), len(output_classes)))

        for epoch in range(len(self.path_history)):
            epoch_paths = self.path_history[epoch]
            for path in epoch_paths:
                path_index = output_classes.index(path)
                path_counts[epoch, path_index] += 1

        ax6 = fig.add_subplot(gs[2, 1])
        sns.heatmap(path_counts, cmap="YlGnBu", xticklabels=output_classes, ax=ax6)
        ax6.set_title(
            f"Path History Heatmap - All Epochs\n(r: {learning_config['rewardSetup']}, p: {learning_config['init_punish']}, ep: {epsilon}, exp_rate: {init_explore_rate:.5f} - {explore_rate:.5f}, exp_times: {exp_counter})")
        ax6.set_xlabel('Output Classes')
        ax6.set_ylabel('Epochs')

        # Plot for device usage history in a single row
        ax7 = fig.add_subplot(gs[3, :])  # Span across both columns
        colors = ['blue'] *  devices_config['iot']['num_devices'] + ['orange'] *  devices_config['mec']['num_devices'] + ['green'] * devices_config['cloud']['num_devices']
        ax7.bar(range(1, len(self.device_usage) + 1), self.device_usage, color=colors)
        ax7.set_title('PE ACTIVITY History')
        ax7.set_xlabel('Device')
        ax7.set_yticks([]) 
        ax7.set_xticks(range(1, len(self.device_usage) + 1)) 

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(learning_config['result_plot_path'])

    def update(self, time_epoch, energy_epoch, reward_epoch, loss_epoch, fail_epoch, usage_epoch, num_episodes,
               path_job,device_usuage):

        avg_time = time_epoch / num_episodes
        avg_energy = energy_epoch / num_episodes
        avg_reward = reward_epoch / num_episodes
        avg_loss = loss_epoch / num_episodes
        avg_fail = [elem / num_episodes for elem in fail_epoch]

        avg_loss = avg_loss.detach().numpy()
        self.avg_loss_history = np.append(self.avg_loss_history, avg_loss)
        self.avg_reward_history = np.append(self.avg_reward_history, avg_reward)
        self.avg_time_history = np.append(self.avg_time_history, avg_time)
        self.avg_energy_history = np.append(self.avg_energy_history, avg_energy)
        self.avg_fail_history = np.vstack([self.avg_fail_history, avg_fail])

        self.iot_usage.append(usage_epoch[0])
        self.mec_usage.append(usage_epoch[1])
        self.cc_usage.append(usage_epoch[2])
        self.path_history.append(path_job)
        self.device_usage = device_usuage
