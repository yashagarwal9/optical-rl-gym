from typing import Tuple

import gym
from gym import spaces
import numpy as np

from .rmsa_env import RMSAEnv
from ..utils import custom_print


class CustomDeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        seed=None,
        allow_rejection=False,
        max_num_nodes=1000,
    ):
        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=False
        )

        self.j = j
        self.max_num_nodes = max_num_nodes
        # self.observation_space = spaces.Dict(
        #     spaces={
        #         "adjacency_matrix": spaces.Box(low=0, high=1, shape=(self.max_num_nodes, self.max_num_nodes), dtype=np.int8),
        #         "node_features": spaces.Box(low=0, high=1, shape=(self.max_num_nodes, self.k_paths), dtype=np.int8),
        #         "traffic_src": spaces.Box(low=0, high=1, shape=(self.max_num_nodes, ), dtype=np.int8),
        #         "traffic_dst": spaces.Box(low=0, high=1, shape=(self.max_num_nodes, ), dtype=np.int8),
        #         "bit_rate": spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
        #         "spectrum_details": spaces.Box(low=-1e6, high=1e6, shape=((2 * self.j + 3) * self.k_paths,), dtype=np.float32),
        #     }
        # )

        shape = (
            (max_num_nodes * max_num_nodes) + (max_num_nodes * self.k_paths) + max_num_nodes + max_num_nodes + 1 + (2 * self.j + 3) * self.k_paths
        )

        self.observation_space = gym.spaces.Box(
            low=-2**30, high=2**30, dtype=np.float64, shape=(shape,)
        )

        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.j + self.reject_action
        )
        self.action_space.seed(self.rand_seed)
        self.seed(self.rand_seed)
        self.reset(only_episode_counters=False)

    def step(self, action: int):
        if action < self.k_paths * self.j:  # action is for assigning a route
            route, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route)
            if block < len(initial_indices):
                return super().step([route, initial_indices[block]])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources])
        else:
            return super().step([self.k_paths, self.num_spectrum_resources])

    def observation(self):
        # TODO Need to construct all these values. They are sampled randomly now

        adjacency_matrix = np.random.randint(2, size=(self.max_num_nodes, self.max_num_nodes)).reshape(1, self.max_num_nodes * self.max_num_nodes)
        node_features = np.random.randint(2, size=(self.max_num_nodes, self.k_paths)).reshape(1, self.max_num_nodes * self.k_paths)
        traffic_src = np.random.randint(2, size=(self.max_num_nodes, )).reshape(1, self.max_num_nodes)
        traffic_dst = np.random.randint(2, size=(self.max_num_nodes, )).reshape(1, self.max_num_nodes)
        bit_rate = np.random.uniform(low=-1e6, high=1e6, size=(1,)).reshape(1, 1)
        spectrum_details = np.random.uniform(low=-1e6, high=1e6, size=((2 * self.j + 3) * self.k_paths,)).reshape(1, (2 * self.j + 3) * self.k_paths)

        # [0 - 999999] + [1000000 - 1004999] + [1005000 - 1005999] + [1006000 - 1006999] + [1007000] + [1007001 - 1007025] => 1007026

        flattened_obs = np.concatenate(
            (
                adjacency_matrix,
                node_features,
                traffic_src,
                traffic_dst,
                bit_rate,
                spectrum_details,
            ),
            axis=1,
        ).reshape(self.observation_space.shape).astype(np.float32)

        return flattened_obs

    def reward(self):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // self.j
        block = action % self.j
        return route, block


def shortest_path_first_fit(env: CustomDeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: CustomDeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
