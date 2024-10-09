import numpy as np

from config import learning_config
from config import jobs_config




class Utility:
    def __init__(self,devices):
        self.devices=devices
        self.min_time, self.max_time, self.min_energy, self.max_energy = self.get_min_max_time_energy()
    
    # FEATURE EXTRACTION
    def get_input(self,task):
    
        if learning_config['regularize_input'] :
            compLoad = [min(jobs_config["task"]["computational_load"])*1e6, max(jobs_config["task"]["computational_load"])*1e6]
            inputs = [min(jobs_config["task"]["input_size"])*1e6, max(jobs_config["task"]["input_size"])*1e6]
            outputs = [min(jobs_config["task"]["output_size"])*1e6, max(jobs_config["task"]["output_size"])*1e6]
            task_features = [
                (task["computational_load"]-compLoad[0])/(compLoad[1]-compLoad[0]),
                (task["input_size"]-inputs[0])/(inputs[1]-inputs[0]),
                (task["output_size"]-outputs[0])/(outputs[1]-outputs[0]),
                task["is_safe"],
            ]
        else:
            task_features = [
                task["computational_load"],
                task["input_size"],
                task["output_size"],
                task["is_safe"],
            ]
            
        if learning_config['onehot_kind']:
            task_features.extend([
                1 if task["task_kind"]==1 else 0,
                1 if task["task_kind"]==2 else 0,
                1 if task["task_kind"]==3 else 0,
                1 if task["task_kind"]==4 else 0,
            ])
        else:
            task_features.extend([
                task["task_kind"],
            ])
        return task_features

           
    def get_min_max_time_energy(self):
        min_time = float('inf')
        max_time = float('-inf')
        min_energy = float('inf')
        max_energy = float('-inf')
        
    
        # Use the smallest and largest computational loads
        for device in self.devices:
            for core_index,core in enumerate(device['voltages_frequencies']):
                for dvfs_index,dvfs in enumerate(device['voltages_frequencies'][core_index]):
                    compLoad = [min(jobs_config["task"]["computational_load"])*1e6, max(jobs_config["task"]  ["computational_load"])*1e6]
                    inputs = [min(jobs_config["task"]["input_size"])*1e6, max(jobs_config["task"]["input_size"])*1e6]
                    outputs = [min(jobs_config["task"]["output_size"])*1e6, max(jobs_config["task"]["output_size"])*1e6]
                    for load in compLoad:
                        for input in inputs:
                            for output in outputs:
                                task = {"computational_load": load,
                                        "input_size": input,
                                        "output_size":output}
                                
                                # Calculate execution time and energy
                                total_time, total_energy = calc_total(device, task,core_index,dvfs_index)
                
                                min_time = min(min_time, total_time)
                                max_time = max(max_time, total_time)
                                min_energy = min(min_energy, total_energy)
                                max_energy = max(max_energy, total_energy)
        return min_time, max_time, min_energy, max_energy


    def getBatteryPunish(self,b_start, b_end, alpha=-100.0, beta=3.0, gamma=0.1):
        if b_start < b_end:
            raise ValueError("Final battery level must be less than or equal to the initial battery level.")

        # Calculate the percentage of battery drained and apply non-linearity
        battery_drain = (b_start - b_end) ** gamma

        # Calculate the exponential penalty factor based on the remaining battery level
        low_battery_factor = ((100 - b_end) / 100) ** beta

        # Calculate the total penalty
        penalty = alpha * battery_drain * low_battery_factor

        return penalty


    def checkBatteryDrain(self,energy, device):
        punish = 0
        batteryFail = 0

        if device['type'] == "iot":
            battery_capacity = device["battery_capacity"]
            battery_start = device['live_state']["battery_now"]
            battery_end = ((battery_start * battery_capacity) - (energy * 1e5)) / battery_capacity
            if battery_end < device["ISL"] * 100:
                batteryFail = 1
                print(device["type"], "battery fail")
            else:
                punish = self.getBatteryPunish(battery_start, battery_end, alpha=learning_config["init_punish"])
                device['live_state']["battery_now"]=battery_end

        return punish, batteryFail


    def regularize_output(self,total_t=0,total_e=0):
        if total_e:
            return (total_e-self.min_energy)/(self.max_energy-self.min_energy)
        if total_t:
            return (total_t-self.min_time)/(self.max_time-self.min_time)
 
    def extract_all_pe_features(self):
        pe_features =[]
        for pe in self.devices:
            pe_features.extend(extract_pe_data(pe))
        return pe_features
    
def extract_pe_data(pe):
    # TODO : regularize
    # devicePower = 0
    # for index, core in enumerate(pe["voltages_frequencies"]):
    #     corePower = 0
    #     for mod in core:
    #         freq, vol = mod
    #         corePower += (freq / vol)
    #     devicePower += corePower

    battery_now = pe['live_state']['battery_now']
    acceptable_tasks = [0, 0, 0, 0]
    for i in range(1, 5):
        if i in pe['acceptable_tasks']:
            acceptable_tasks[i - 1] = 1

    # return [battery_now /100, pe['is_safe']] + acceptable_tasks
    return [battery_now /100, pe['is_safe']] 


# FORMULAS
def calc_execution_time(device, task, core, dvfs):
    return task["computational_load"] / device["voltages_frequencies"][core][dvfs][0]


def calc_power_consumption(device, task, core, dvfs):
    if device['type'] == "cloud":
        return device["voltages_frequencies"][core][dvfs][1]
    return (device["capacitance"] * (device["voltages_frequencies"][core][dvfs][1] ** 2) *
            device["voltages_frequencies"][core][dvfs][0])


def calc_energy(device, task, core, dvfs):
    return calc_execution_time(device, task, core, dvfs) * calc_power_consumption(device, task, core, dvfs)


def calc_total(device, task, core, dvfs):
    timeTransMec = 0
    timeTransCC = 0
    baseTime = 0
    baseEnergy = 0
    totalEnergy = 0
    totalTime = 0

    transferRate5g = 1e9
    latency5g = 5e-3
    transferRateFiber = 1e10
    latencyFiber = 1e-3
    timeDownMec = task["output_size"] / transferRate5g
    timeDownMec += latency5g
    timeUpMec = task["input_size"] / transferRate5g
    timeUpMec += latency5g

    alpha = 52e-5
    beta = 3.86412
    powerMec = alpha * 1e9 / 1e6 + beta

    timeDownCC = task["output_size"] / transferRateFiber
    timeDownCC += latencyFiber
    timeUpCC = task["input_size"] / transferRateFiber
    timeUpCC += latencyFiber

    powerCC = 3.65

    if device["type"] == "mec":
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec
        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime + timeTransMec
        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy + energyTransMec

    elif device["type"] == "cloud":
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec

        timeTransCC = timeUpCC + timeDownCC
        energyTransCC = powerCC * timeTransCC

        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime + timeTransMec + timeTransCC

        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy + energyTransMec + energyTransCC

    elif device["type"] == "iot":
        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime
        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy

    return totalTime, totalEnergy


# REWARDS AND PUNISHMENTS
def reward_function(e=0, t=0, punish=0):
    setup = learning_config['rewardSetup']
    alpha = learning_config['alpha']
    beta = learning_config['beta']

    if punish and learning_config['increasing_punish']:
        learning_config['init_punish'] += learning_config['punish_epsilon']

    if punish:
        return learning_config['init_punish']

    if setup == 1:
        return -1 * (alpha * e + beta * t)
    elif setup == 2:
        return 1 / (alpha * e + beta * t)
    elif setup == 3:
        return -np.exp(alpha * e) - np.exp(beta * t)
    elif setup == 4:
        return -np.exp(alpha * e + beta * t)
    elif setup == 5:
        return np.exp(-1 * (alpha * e + beta * t))
    elif setup == 6:
        return -np.log(alpha * e + beta * t)
    elif setup == 7:
        return -((alpha * e + beta * t) ** 2)

