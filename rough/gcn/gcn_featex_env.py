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

# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
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

# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
topology_name = 'nsfnet_chen'
k_paths = 5

with open(f'/nobackup/kpvsms/myenv/optical-rl-gym/examples/topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
    topology = pickle.load(f)


monitor_info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate', 'service_blocking_rate', 'bit_rate_blocking_rate', 'network_compactness', 'network_compactness_difference', 'avg_link_compactness', 'avg_link_utilization')


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
                episode_length=101, node_request_probabilities=node_request_probabilities,
                max_num_nodes=1000)
topology.edges()
# # Create log dir
log_dir = "/nobackup/kpvsms/myenv/optical-rl-gym/logs/CustomDeepRMSA-a2c/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, show_plot=False)

env = gym.make('CustomDeepRMSA-v0', **env_args)
print("Checking env...")
# check_env(env)
# logs will be saved in log_dir/training.monitor.csv
# in this case, on top of the usual monitored things, we also   monitor service and bit rate blocking rates
env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)
# for more information about the monitor, check https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/bench/monitor.html#Monitor

# here goes the arguments of the policy network to be used
policy_args = dict(
	net_arch=[2538, 1024, 512, 256, 128],
    features_extractor_class=CustomGCNGRUFeatureExtractor,
	features_extractor_kwargs={"features_dim": 2538, "max_num_nodes": 1000, "num_k_paths": 5, "batch_size": 1}
)

agent = A2C(
    CustomPolicy,
	env,
	verbose=1,
	tensorboard_log="/nobackup/kpvsms/myenv/optical-rl-gym/tb/A2C-CustomDeepRMSA-v0/",
	policy_kwargs=policy_args,
	gamma=.95,
	learning_rate=10e-6,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

agent.learn(total_timesteps=100000, callback=callback)

