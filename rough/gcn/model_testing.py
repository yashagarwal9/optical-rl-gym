from torch.optim.rmsprop import RMSprop
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import h5py
from IPython.display import clear_output
import torch
import gym
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common import results_plotter

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1, show_plot: bool=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.show_plot = show_plot

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.show_plot and self.n_calls % self.check_freq == 0 and self.n_calls > 5001:
            plotting_average_window = 100

            training_data = pd.read_csv(self.log_dir + 'training.monitor.csv', skiprows=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

            ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')

            ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode service blocking rate')

            ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode bit rate blocking rate')

            # fig.get_size_inches()
            plt.tight_layout()
            plt.show()

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True

from rough.code.custom_gcn_gru_policy import CustomPolicy
from rough.code.custom_gcn_gru_feature_extractor import CustomGCNGRUFeatureExtractor
from stable_baselines3.common.env_checker import check_env


topology_name = 'nsfnet_chen'
k_paths = 5

with open(f'/nobackup/kpvsms/myenv/optical-rl-gym/examples/topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
    topology = pickle.load(f)


monitor_info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate')


print(topology)
# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
                                       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
                                       0.07607608, 0.12012012, 0.01901902, 0.16916917])

# mean_service_holding_time=7.5,
env_args = dict(topology=topology, seed=10,
                allow_rejection=False, # the agent cannot proactively reject a request
                j=1, # consider only the first suitable spectrum block for the spectrum assignment
                mean_service_holding_time=7.5, # value is not set as in the paper to achieve comparable reward values
                episode_length=50, node_request_probabilities=node_request_probabilities,
                max_num_nodes=1000)
topology.edges()
# # Create log dir
log_dir = "/nobackup/kpvsms/myenv/optical-rl-gym/logs/CustomDeepRMSA-a2c/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, show_plot=False)

env = gym.make('CustomDeepRMSA-v0', **env_args)

custom_objects = {
"observation_space": env.observation_space,
"action_space": env.action_space,
"policy_class": CustomPolicy
}

loaded_model = A2C.load(
"/nobackup/kpvsms/myenv/optical-rl-gym/logs/CustomDeepRMSA-a2c/best_model.zip",
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
custom_objects=custom_objects,
policy_kwargs={'net_arch': [2538, 1024, 512, 256, 128], 'features_extractor_class': CustomGCNGRUFeatureExtractor, 'features_extractor_kwargs': {'features_dim': 2538, 'max_num_nodes': 1000, 'num_k_paths': 5, 'batch_size': 1}, 'optimizer_class': RMSprop, 'optimizer_kwargs': {'alpha': 0.99, 'eps': 1e-05, 'weight_decay': 0}},
print_system_info=True)

import time

obs = env.reset()
rewards = None
dones = None
info = None
action = None
_states = None
actions = []
dones_arr = []
ct=1
env.render()
for _ in range(100):
    action, _states = loaded_model.predict(obs, deterministic=True)
    print(ct)
    ct+=1
    actions.append(action)
    obs, rewards, dones, info = env.step(action)
    dones_arr.append(dones) 

print(actions)


