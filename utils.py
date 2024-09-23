import numpy as np
def get_input(task, pe_dict={},device_features=False,subtree=False):
        task_features = [
            task["computational_load"],
            task["input_size"],
            task["output_size"],
            task["kind1"],
            task["kind2"],
            task["kind3"],
            task["kind4"],
            task["is_safe"],
        ]
        if not subtree and not device_features:
            return task_features
        
        pe_features = []
        for pe in pe_dict.values():
            pe_features.extend(get_pe_data(pe, pe['id'],subtree))
        return task_features + pe_features

def get_pe_data(pe_dict, pe_id,subtree):
        pe = None
        # state.database.get_device(pe_id)
        devicePower = pe['devicePower']

        batteryLevel = pe_dict['batteryLevel']
        battery_capacity = pe['battery_capacity']
        battery_isl = pe['ISL']
        battery = ((1 - battery_isl) * battery_capacity - batteryLevel) / battery_capacity

        num_cores = pe['num_cores']
        cores = 1 - (sum(pe_dict['occupiedCores']) / num_cores)
        if subtree:
            return pe_dict['occupiedCores'] + [ devicePower, battery]
        else:
            return [cores, devicePower, battery]

def reward_function(setup=5, e=0, alpha=1, t=0, beta=1, punish=0):
    if punish:
        return -10

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