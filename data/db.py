from data.gen import Generator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Database, cls).__new__(cls, *args, **kwargs)
            cls._instance._devices = Generator.get_devices()
            cls._instance._jobs, cls._instance._tasks = Generator.get_jobs()
            cls._instance._task_norm = cls._instance.normalize_tasks(cls._instance._tasks.copy())
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


        