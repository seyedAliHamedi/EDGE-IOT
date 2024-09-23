import numpy as np
import pandas as pd
from typing import Counter
from sklearn.cluster import KMeans


def balance_kmeans_cluster(devices, k=2):
        data = [get_pe_data(device) for _, device in devices.iterrows()]

        if len(devices) < k + 1:
            return [devices] * k
        X = np.array(data)
        kmeans = KMeans(n_clusters=k, init="random", random_state=42)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        clusters = [devices.iloc[[]].copy() for _ in range(k)]

        # Balance the clusters
        balanced_labels = balance_clusters(cluster_labels, k, len(devices))

        for (index, device), label in zip(devices.iterrows(), balanced_labels):
            clusters[label] = pd.concat([clusters[label], device.to_frame().T], ignore_index=True)
        return clusters

def get_pe_data(pe):
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
            devicePower = devicePower / pe['number_of_cpu_cores']
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