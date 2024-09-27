import itertools
import pandas as pd
from env.env import Environment

# Import the configs (jobs_config, devices_config, learning_config)
from config import jobs_config, devices_config, learning_config

# Define parameters for grid search in lists
learning_algorithms = ['policy-grad', 'a2c', 'ppo']
tree_max_depths = [2, 3, 4]
reward_setups = [1, 5, 7]


# Create a grid of all combinations of parameters
grid = itertools.product(learning_algorithms, tree_max_depths, reward_setups)
results = []

if __name__ == '__main__':
    for i, (learning_algorithm, tree_max_depth, reward_setup) in enumerate(grid):
        # Update the learning_config
        learning_config['learning_algorithm'] = learning_algorithm
        learning_config['tree_max_depth'] = tree_max_depth
        learning_config['rewardSetup'] = reward_setup

        print(f"Running configuration {i + 1} with:")
        print(f"  - learning_algorithm: {learning_algorithm}")
        print(f"  - tree_max_depth: {tree_max_depth}")
        print(f"  - rewardSetup: {reward_setup}")

        # Run the environment
        env = Environment()
        result = env.run()

        # Save results (assuming env.run() returns some performance metric or result)
        results.append({
            'learning_algorithm': learning_algorithm,
            'tree_max_depth': tree_max_depth,
            'reward_setup': reward_setup,
            'result': result  # Store the outcome from the environment
        })

    # Save the results to a CSV file
    pd.DataFrame(results).to_csv('grid_search_results.csv', index=False)
