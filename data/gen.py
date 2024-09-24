import pandas as pd
import numpy as np
import os
import ast
from data.config import devices_config, jobs_config
import networkx as nx
import random


class Generator:
    _task_id_counter = 0
    _device_id_counter = 0
    _job_id_counter = 0

    _devices_path = os.path.join(os.path.dirname(__file__), "resources", "devices.csv")
    _job_path = os.path.join(os.path.dirname(__file__), "resources", "jobs.csv")
    _tasks_path = os.path.join(os.path.dirname(__file__), "resources", "tasks.csv")

    @classmethod
    def get_devices(cls, file_path=_devices_path):
        if os.path.exists(file_path):
            devices = pd.read_csv(file_path)
            devices["voltages_frequencies"] = devices["voltages_frequencies"].apply(
                lambda x: ast.literal_eval(x))
            devices["capacitance"] = devices["capacitance"].apply(
                lambda x: ast.literal_eval(x)
            )
            devices["powerIdle"] = devices["powerIdle"].apply(
                lambda x: ast.literal_eval(x)
            )
            devices["acceptableTasks"] = devices["acceptableTasks"].apply(
                lambda x: ast.literal_eval(x)
            )
            return devices
        else:
            return Generator._generate_device()

    # generate random Processing element attributes based on the bounds and ranges defined in the config

    #   frequency in KHZ
    #   voltage in Volt
    #   capacitance in nano-Farad
    #   powerIdle in Watt
    #   ISL in percentage
    #   battery capacity in W*micro-second : 36000 Ws - Equivalent to 36000*10^3 W*milli-second, 10Wh or

    @classmethod
    def _generate_device(cls, config=devices_config, file_path=_devices_path):
        devices_data = []
        for type in ("iot", "mec", "cloud"):
            config = devices_config[type]

            for _ in range(config["num_devices"]):
                cpu_cores = (
                    config['num_cores']
                    if type == 'cloud'
                    else int(np.random.choice(config["num_cores"]))
                )
                device_info = {
                    "id": Generator._device_id_counter,
                    "type": type,
                    "num_cores": cpu_cores,
                    "voltages_frequencies": [
                        [
                            config["voltage_frequencies"][i]
                            for i in np.random.choice(len(config['voltage_frequencies']), size=3, replace=False)
                        ]
                        for _ in range(cpu_cores)
                    ],
                    "ISL": (
                        -1
                        if config["isl"] == -1
                        else np.random.uniform(config["isl"][0], config["isl"][1])
                    ),
                    "capacitance": [
                        np.random.uniform(
                            config["capacitance"][0], config["capacitance"][1]
                        )
                        * 1e-9
                        for _ in range(cpu_cores)
                    ],
                    "powerIdle": [
                        float(np.random.choice(config["powerIdle"])) * 1e-6
                        for core in range(cpu_cores)
                    ],
                    "battery_capacity": (
                        -1
                        if config["battery_capacity"] == -1
                        else np.random.uniform(
                            config["battery_capacity"][0], config["battery_capacity"][1]
                        ) * 1e3
                    ),
                    "battery_level": -1 if config["battery_capacity"] == -1 else 100,
                    "error_rate": np.random.uniform(
                        config["error_rate"][0], config["error_rate"][1]
                    ),
                    "acceptableTasks": list(np.random.choice(
                        jobs_config["task"]["task_kinds"],
                        size=np.random.randint(config["num_acceptable_task"][0], config["num_acceptable_task"][1]),
                        replace=False,
                    )),
                    "handleSafeTask": int(
                        np.random.choice(
                            [0, 1], p=[config["safe"][0], config["safe"][1]]
                        )
                    ),
                    "maxQueue": config["maxQueue"],

                }
                if type == 'cloud':
                    device_info['acceptableTasks'] = [1, 2, 3, 4]
                Generator._device_id_counter += 1
                devices_data.append(device_info)

        devices = pd.DataFrame(devices_data)
        devices['name'] = devices.apply(
            lambda row: row['type'] + str(row.name), axis=1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        devices.to_csv(file_path, index=False)
        return devices

    @classmethod
    def get_jobs(cls, file_path_jobs=_job_path, file_path_tasks=_tasks_path):
        if os.path.exists(file_path_jobs):
            jobs = pd.read_csv(file_path_jobs)
            jobs["tasks_ID"] = jobs["tasks_ID"].apply(lambda x: ast.literal_eval(x))
            jobs["tree"] = jobs["tree"].apply(lambda x: ast.literal_eval(x))
            tasks = pd.read_csv(file_path_tasks)

            tasks["predecessors"] = tasks["predecessors"].apply(lambda x: ast.literal_eval(x))
            # tasks["successors"] = tasks["successors"].apply(lambda x: ast.literal_eval(x))
            return jobs, tasks
        else:
            return Generator._generate_jobs()

    @classmethod
    def _generate_jobs(cls, config=jobs_config, file_path_jobs=_job_path, file_path_tasks=_tasks_path):
        max_deadline = config["max_deadline"]
        task_config = config["task"]
        tasks_data = []
        jobs_data = []
        start_node_number = 1  # Keep track of task IDs across DAGs

        for job_id in range(config["num_jobs"]):
            # Generate a random DAG for the job
            num_nodes = random.randint(config["min_num_nodes_dag"], config["max_num_nodes_dag"])
            random_dag = cls.generate_random_dag(num_nodes)

            # Map node IDs to the continuous task IDs
            mapping = {f"t{i}": f"t{i + start_node_number - 1}" for i in range(1, num_nodes + 1)}
            random_dag = nx.relabel_nodes(random_dag, mapping)

            # Create tasks from the nodes in the DAG
            for node in random_dag.nodes:
                parents = list(random_dag.predecessors(node))
                task_info = {
                    "id": Generator._task_id_counter,
                    "job_id": job_id,
                    "predecessors": parents,
                    "pred_count": len(parents),
                    "computational_load": np.random.randint(task_config["computational_load"][0],
                                                            task_config["computational_load"][1]) * 1e6,
                    "input_size": np.random.randint(task_config["input_size"][0], task_config["input_size"][1]) * 1e6,
                    "output_size": np.random.randint(task_config["output_size"][0],
                                                     task_config["output_size"][1]) * 1e6,
                    "task_kind": np.random.choice(task_config["task_kinds"]),
                    "is_safe": np.random.choice([0, 1],
                                                p=[task_config["safe_measurement"][0],
                                                   task_config["safe_measurement"][1]]),
                    "is_head": len(parents) == 0,  # A head node has no predecessors
                    "is_tail": len(list(random_dag.successors(node))) == 0,  # A tail node has no successors
                    "isReady": 0,
                }
                tasks_data.append(task_info)
                Generator._task_id_counter += 1

            # Update task IDs for the next job
            start_node_number += num_nodes

            # Create job information
            job_info = {
                "id": job_id,
                "task_count": len(random_dag.nodes),
                "tasks_ID": [task_info["id"] for task_info in tasks_data if task_info["job_id"] == job_id],
                "tree": [(task_info["id"], task_info["predecessors"]) for task_info in tasks_data if
                         task_info["job_id"] == job_id],
                "deadline": np.random.randint(1, max_deadline),
            }
            jobs_data.append(job_info)

        # Save jobs and tasks
        jobs = pd.DataFrame(jobs_data)
        tasks = pd.DataFrame(tasks_data)
        jobs.to_csv(file_path_jobs, index=False)
        tasks.to_csv(file_path_tasks, index=False)
        return jobs, tasks

    @staticmethod
    def generate_random_dag(num_nodes, max_num_parents_dag=3):
        """Generates a random Directed Acyclic Graph (DAG) with a specified number of nodes."""
        dag = nx.DiGraph()
        nodes = [f"t{i + 1}" for i in range(num_nodes)]
        dag.add_nodes_from(nodes)

        # Create parent-child relationships (DAG)
        available_parents = {node: list(nodes[:i]) for i, node in enumerate(nodes)}
        for i in range(2, num_nodes + 1):
            num_parents = min(random.randint(1, min(i, max_num_parents_dag)), len(available_parents[f"t{i}"]))
            parent_nodes = random.sample(available_parents[f"t{i}"], num_parents)
            dag.add_edges_from((parent_node, f"t{i}") for parent_node in parent_nodes)
            available_parents[f"t{i}"] = list(nodes[:i])

        return dag
