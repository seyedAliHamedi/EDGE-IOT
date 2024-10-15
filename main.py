from env.env import Environment
from config import learning_config
from model.actor_critic import ActorCritic
from model.agent import Agent
from model.shared_adam import SharedAdam
import torch.multiprocessing as mp

if __name__ == '__main__':
    barrier = mp.Barrier(learning_config['multi_agent'] + 1)
    env = Environment(barrier)
     # define the global Actor-Critic and the shared optimizer (A3C)
    global_actor_critic = ActorCritic(devices=env.devices)
    global_actor_critic.share_memory()
    global_optimizer = SharedAdam(global_actor_critic.parameters())
    # setting up workers and their barriers
    workers = []
    # kick off the state
    print("Simulation starting...")
    for i in range(learning_config['multi_agent']):
        # organize and start the agents
        worker = Agent(name=f'worker_{i}', global_actor_critic=global_actor_critic,
                       global_optimizer=global_optimizer, barrier=barrier,env=env)
        workers.append(worker)
        worker.start()
    print("Simulation starting2 ...")
    env.run()