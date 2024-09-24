devices_config = {
    "iot": {
        "num_devices": 10,
        "num_cores": [4],
        "voltage_frequencies": [
            (10e6, 1.8),
            (20e6, 2.3),
            (40e6, 2.7),
            (80e6, 4.0),
            (160e6, 5.0),
        ],
        "isl": (0.1, 0.2),
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (0.2, 0.3),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [800, 900, 1000],
        # battery_capacity in Watt-second
        "battery_capacity": (36, 41),
        "error_rate": (0.01, 0.06),
        "safe": (0.25, 0.75),
        "maxQueue": 5
    },
    "mec": {
        "num_devices": 5,
        "num_cores": [8],
        "voltage_frequencies": [
            (600 * 1e6, 0.8),
            (750 * 1e6, 0.825),
            (1000 * 1e6, 1.0),
            (1500 * 1e6, 1.2),
        ],
        "isl": -1,
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (1.5, 2),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [550000, 650000, 750000],
        "battery_capacity": -1,
        "error_rate": (0.5, 0.11),
        "safe": (0.5, 0.5),
        "maxQueue": 1

    },
    "cloud": {
        "num_devices": 1,
        "num_cores": 128,
        # TODO cloud , please correct these numbers
        "voltage_frequencies": ((2.8e9, 13.85e-6), (3.9e9, 24.28e-6), (5e9, 36e-6)),
        "isl": -1,
        "capacitance": (3, 5),
        "powerIdle": [0],
        "battery_capacity": -1,
        "error_rate": (0.10, 0.15),
        "safe": (1,0),
        "maxQueue": 1
    },
}

jobs_config = {
    "num_jobs": 1000,
    "max_deadline": 2000,
    "max_task_per_depth": 2,
    "max_depth": 2,
    "task": {
        # input_size ,output_size ,computational_load in MB --> * 10^6
        "input_size": [1, 1001],
        "output_size": [1, 1001],
        "computational_load": [1, 1001],
        "safe_measurement": [0.8, 0.2],
        "task_kinds": [1, 2, 3, 4],
    },
}



learning_config={
    "tree":"soft-ddt", # ddt
    
    #    ddt :  ddt 
    #    device-ddt :  ddt + pe features
    #    soft-ddt :  soft tree
    #    soft-device-ddt :  soft tree + pe feature
    #    clustree : clustree 
    #    device-clustree : clustree + pe features
    
    "rewardSetup":5, 
       
    #   1 : -1 * (alpha * e + beta * t)
    #   2 : 1 / (alpha * e + beta * t)
    #   3 : -np.exp(alpha * e) - np.exp(beta * t)
    #   4 : -np.exp(alpha * e + beta * t)
    #   5 : np.exp(-1 * (alpha * e + beta * t))
    #   6 : -np.log(alpha * e + beta * t)
    #   7 : -((alpha * e + beta * t) ** 2)
    
    "alpha":1, # energy coffeicent in the reware
    "beta":1, # time coffeicent in the reware
    
    "increasing_punish":True,
    "init_punish":-10,
    "punish_epsilon":-0.001,
    
    "should_explore":True,
    "explore_epsilon":True,
    "num_jobs":1000,
    
    "tree_max_depth":3,

 
    "onehot_kind":True,   # onehotting the task kind
    "regulrize_input":True, # regulirze the task/device features to feed the tree
    "regulrize_output":True,  # regulirze t,e for the reward
    
    
    'pe_num_features':2,
    
    
    
    "learning_algorithm":"ppo",
    "learning_algorithm":"a2c",
    "learning_algorithm":"ploicy-grad",
    "discount_factor":0.99, # 0: reward , 0.99:return
    
    "result_summery_path":'./results/summery.csv',
    "result_plot_path":'./results/result.png',
    
    
}