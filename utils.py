import numpy as np
from typing import Counter
from sklearn.cluster import KMeans
from data.config import learning_config
from data.db import Database
from collections import Counter


# FEATURE EXTRACTION
def get_input(task):
    if learning_config['onehot_kind']:
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
    else:
        task_features = [
            task["computational_load"],
            task["input_size"],
            task["output_size"],
            task["task_kind"],
            task["is_safe"],
        ]
    pe_features = []
    if learning_config['tree'] in ("device-clustree", "soft-device-ddt", "device-ddt"):
        for pe in Database().get_all_devices():
            pe_features.extend(extract_pe_data(pe))
    return task_features + pe_features


def extract_pe_data(pe):
    if pe['type'] != "cloud":
        devicePower = 0
        for index, core in enumerate(pe["voltages_frequencies"]):
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / pe['num_cores']
    else:
        devicePower = 1e9

    batteryLevel = pe['battery_level']
    battery_capacity = pe['battery_capacity']
    battery_isl = pe['ISL']
    battery = ((1 - battery_isl) * battery_capacity - batteryLevel) / battery_capacity

    return [devicePower, battery]


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


# FORMULAS
def calc_execution_time(device, task, core, dvfs):
    if device['id'] == "cloud":
        return task["computational_load"] / device["voltages_frequencies"][0]
    else:
        return task["computational_load"] / device["voltages_frequencies"][core][dvfs][0]


def calc_power_consumption(device, task, core, dvfs):
    if device['type'] == "cloud":
        return device["voltages_frequencies"][core][dvfs][1]
    return (device["capacitance"][core] * (device["voltages_frequencies"][core][dvfs][1] ** 2) *
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


# CLUSTERING
def balance_kmeans_cluster(devices, k=2, random_state=42):
    data = [extract_pe_data_for_clustering(device) for device in devices]
    if len(devices) < k:
        return [devices] * k
    X = np.array(data)
    kmeans = KMeans(n_clusters=k, init="random", random_state=random_state)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    clusters = [[] for _ in range(k)]

    balanced_labels = balance_clusters(cluster_labels, k, len(devices))

    for device, label in zip(devices, balanced_labels):
        clusters[label].append(device)
    return clusters


def extract_pe_data_for_clustering(pe):
    capacitance = sum(pe['capacitance'])
    handleSafeTask = pe['handleSafeTask']
    kind = sum(pe['acceptableTasks'])

    if pe['id'] != "cloud":
        devicePower = 0
        for index, core in enumerate(pe["voltages_frequencies"]):
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / pe['num_cores']
    else:
        devicePower = 1e9

    return [devicePower, capacitance, handleSafeTask, kind]


def balance_clusters(labels, k, n_samples):
    """
        Adjusts the initial cluster assignments to ensure clusters are balanced.
        """
    target_cluster_size = n_samples // k
    max_imbalance = n_samples % k  # Allowable imbalance due to indivisible n_samples

    cluster_sizes = Counter(labels)

    # List to store the indices of samples in each cluster
    cluster_indices = {i: [] for i in range(k)}

    # Populate the cluster_indices dictionary
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    # Reassign samples to achieve balanced clusters
    for cluster in range(k):
        while len(cluster_indices[cluster]) > target_cluster_size:
            for target_cluster in range(k):
                if len(cluster_indices[target_cluster]) < target_cluster_size:
                    sample_to_move = cluster_indices[cluster].pop()
                    labels[sample_to_move] = target_cluster
                    cluster_indices[target_cluster].append(sample_to_move)
                    # Exit early if target sizes are met with allowable imbalance
                    if _clusters_balanced(cluster_indices, target_cluster_size, max_imbalance):
                        return labels
                    break

    return labels


def _clusters_balanced(cluster_indices, target_size, max_imbalance):
    imbalance_count = sum(abs(len(indices) - target_size) for indices in cluster_indices.values())
    return imbalance_count <= max_imbalance
