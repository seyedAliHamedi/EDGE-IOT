from data.gen import Generator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from config import devices_config,jobs_config

class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Database, cls).__new__(cls, *args, **kwargs)
            cls._instance._devices = Generator.get_devices()
            cls._instance._jobs, cls._instance._tasks = Generator.get_jobs()
            cls._instance._task_norm = cls._instance.normalize_tasks(cls._instance._tasks.copy())
            cls._instance.min_time, cls._instance.max_time, cls._instance.min_energy, cls._instance.max_energy = cls._instance.get_min_max_time_energy()
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None

    # ------------ all ----------

    def get_all_devices(self):
        return self._devices.to_dict(orient='records')

    def get_all_jobs(self):
        return self._jobs.to_dict(orient='records')

    def get_all_tasks(self):
        return self._tasks.to_dict(orient='records')

    # ---------- single ------------

    def get_device(self, id):
        return self._devices.iloc[id].to_dict()

    def get_job(self, id):
        return self._jobs.iloc[id].to_dict()

    def get_task(self, id):
        return self._tasks.iloc[id].to_dict()

    def get_task_norm(self, id):
        return self._task_norm.iloc[id].to_dict()

    # ---------- add & remove ------------
    def add_device(self, id):
        # Generate a new random device
        new_device = Generator.generate_random_device()

        # Convert the new device to a DataFrame if it's not already
        new_device_df = pd.DataFrame([new_device])

        # Concatenate the new device to the existing dataframe
        self._devices = pd.concat([self._devices, new_device_df], ignore_index=True)
        self._devices.reset_index(drop=True, inplace=True)
        self._devices['id'] = self._devices.index
        return new_device

    def remove_device(self, id):
        # Assuming `id` is a column in the devices dataframe
        self._devices = self._devices[self._devices['id'] != id]
        self._devices.reset_index(drop=True, inplace=True)
        self._devices['id'] = self._devices.index

    # -------- normalize -------
    def normalize_tasks(self, tasks_normalize):
        for column in tasks_normalize.columns.values:
            if column in ("computational_load", "input_size", "output_size", "is_safe"):
                tasks_normalize[column] = (tasks_normalize[column] - tasks_normalize[column].min()) / (
                        tasks_normalize[column].max() - tasks_normalize[column].min())
        kinds = [1, 2, 3, 4]
        for kind in kinds:
            tasks_normalize[f'kind{kind}'] = tasks_normalize['task_kind'].isin([kind]).astype(int)
        tasks_normalize.drop(['task_kind'], axis=1)
        return tasks_normalize

    def set_device_battery(self, id, end):
        self._devices.loc[self._devices['id'] == id, 'battery_now'] = end
    def set_core_occupied(self, id, core_i):
        # Locate the row in the DataFrame corresponding to the device by 'id'
        device_index = self._devices.index[self._devices['id'] == id].tolist()

        if device_index:
            device_index = device_index[0]  # Get the index of the first match
            
            # Retrieve the 'occupied_cores' list, making a copy to avoid direct modifications
            occupied_cores = self._devices.at[device_index, 'occupied_cores'].copy()
            
            # Set the specified core to 1 (occupied), ensuring index is within bounds
            if 0 <= core_i < len(occupied_cores):
                occupied_cores[core_i] = 1
                
                # Update the DataFrame with the modified list
                self._devices.at[device_index, 'occupied_cores'] = occupied_cores


    def update_core_occupied(self):
        # Iterate through each device
        for idx, row in self._devices.iterrows():
            # Get the list of occupied cores for the current device
            occupied_cores = row['occupied_cores']
            
            # Increment the value of every occupied core (those not equal to -1)
            updated_cores = [core + 1 if core != -1 else -1 for core in occupied_cores]

            # Set cores back to zero if their value is greater than 5
            updated_cores = [-1 if core > 200 else core for core in updated_cores]
            
            # Update the device's occupied cores
            self._devices.at[idx, 'occupied_cores'] = updated_cores


    def get_min_max_time_energy(self):
        min_time = float('inf')
        max_time = float('-inf')
        min_energy = float('inf')
        max_energy = float('-inf')

        for device_type, device in devices_config.items():
            max_dvfs = max(device['voltage_frequencies'], key=lambda vf: vf[0])
            min_dvfs = min(device['voltage_frequencies'], key=lambda vf: vf[0])
            dvfss = [min_dvfs,max_dvfs]
            compLoad = [min(jobs_config["task"]["computational_load"]), max(jobs_config["task"]["computational_load"])]
            inputs = [min(jobs_config["task"]["input_size"]), max(jobs_config["task"]["input_size"])]
            outputs = [min(jobs_config["task"]["output_size"]), max(jobs_config["task"]["output_size"])]
            # Use the smallest and largest computational loads
            for dvfs in dvfss:
                for load in compLoad:
                    for input in inputs:
                        for output in outputs:
                            task = {"computational_load": load,
                                    "input_size": input,
                                    "output_size":output}
                            
                            # Calculate execution time and energy
                            total_time, total_energy = calc_total(device_type, task,dvfs)
            
                            min_time = min(min_time, total_time)
                            max_time = max(max_time, total_time)
                            min_energy = min(min_energy, total_energy)
                            max_energy = max(max_energy, total_energy)
        return min_time, max_time, min_energy, max_energy





# FORMULAS
def calc_execution_time(task,  dvfs):
        return task["computational_load"] / dvfs[0]


def calc_power_consumption(device_type, task, dvfs):
    if device_type == "cloud":
        return dvfs[1]
    return (1e-9 * dvfs[1] ** 2) *dvfs[0]


def calc_energy(device_type, task, dvfs):
    return calc_execution_time( task,  dvfs) * calc_power_consumption(device_type, task, dvfs)


def calc_total(device_type, task, dvfs):
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

    if device_type== "mec":
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec
        baseTime = calc_execution_time( task,  dvfs)
        totalTime = baseTime + timeTransMec
        baseEnergy = calc_energy( device_type,task,  dvfs)
        totalEnergy = baseEnergy + energyTransMec

    elif device_type == "cloud":
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec

        timeTransCC = timeUpCC + timeDownCC
        energyTransCC = powerCC * timeTransCC

        baseTime = calc_execution_time( task,  dvfs)
        totalTime = baseTime + timeTransMec + timeTransCC

        baseEnergy = calc_energy( device_type,  task, dvfs)
        totalEnergy = baseEnergy + energyTransMec + energyTransCC

    elif device_type == "iot":
        baseTime = calc_execution_time( task, dvfs)
        totalTime = baseTime
        baseEnergy = calc_energy( device_type, task,  dvfs)
        totalEnergy = baseEnergy

    return totalTime, totalEnergy
