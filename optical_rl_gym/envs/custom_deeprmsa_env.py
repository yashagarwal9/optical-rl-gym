from typing import Tuple

import gym
from gym import spaces
import numpy as np

from .rmsa_env import RMSAEnv


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
        self.logger.setLevel("DEBUG")
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
            low=-2 ** 30, high=2 ** 30, dtype=np.float64, shape=(shape,)
        )

        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.j + self.reject_action
        )
        self.action_space.seed(self.rand_seed)
        self.seed(self.rand_seed)
        self.reset(only_episode_counters=False)

    def step(self, action: int):
        if action < self.k_paths * self.j:  # action is for assigning a route
            print(f"    Action: {action}")
            route, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route)
            if block < len(initial_indices):
                return super().step([route, initial_indices[block]])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources])
        else:
            return super().step([self.k_paths, self.num_spectrum_resources])

    def observation(self):

        src_node_id = self.current_service.source_id
        dst_node_id = self.current_service.destination_id
        src_node = self.current_service.source
        dst_node = self.current_service.destination
        k_paths = self.k_shortest_paths[src_node, dst_node]

        # max_num_nodes = Maximum Number of edges supported

        k_paths_adjacency_matrix = np.zeros((self.max_num_nodes, self.max_num_nodes))
        node_features_matrix = np.zeros((self.max_num_nodes, self.k_paths))

        for kpath_num, path in enumerate(k_paths):
            # path_best_modulation = path.best_modulation
            # hops = path.hops
            # total_path_length = path.length
            # path_id = path.path_id
            path_node_list = path.node_list
            for i in range(len(path_node_list) - 1):
                prev_node = path_node_list[i]
                prev_node_id = self.topology.graph["node_indices"].index(prev_node)
                next_node = path_node_list[i + 1]
                next_node_id = self.topology.graph["node_indices"].index(next_node)
                if 0 <= int(prev_node_id) < self.max_num_nodes and 0 <= int(next_node_id) < self.max_num_nodes:
                    k_paths_adjacency_matrix[int(prev_node_id), int(next_node_id)] = 1
                edge_bw_prev_next_node = self.topology.get_edge_data(prev_node, next_node)
                if 0 <= int(edge_bw_prev_next_node["id"]) < self.max_num_nodes and 0 <= int(kpath_num) < self.k_paths:
                    node_features_matrix[int(edge_bw_prev_next_node["id"]), int(kpath_num)] = 1

        traffic_src_obs = np.zeros((self.max_num_nodes,))
        if 0 <= int(src_node) < self.max_num_nodes:
            traffic_src_obs[src_node_id] = 1

        traffic_dst_obs = np.zeros((self.max_num_nodes,))
        if 0 <= dst_node_id < self.max_num_nodes:
            traffic_dst_obs[dst_node_id] = 1

        bit_rate_obs = np.zeros((1, ))
        bit_rate_obs[0] = self.current_service.bit_rate / 100

        spectrum_obs = np.full((self.k_paths, 2 * self.j + 3), fill_value=-1.0)
        for idp, route in enumerate(k_paths):
            available_slots = self.get_available_slots(route)
            num_slots = self.get_number_slots(route)
            initial_indices, lengths = self.get_available_blocks(idp)

            for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                # initial slot index
                spectrum_obs[idp, idb * 2 + 0] = (2 * (initial_index - 0.5 * self.num_spectrum_resources) / self.num_spectrum_resources)
                # number of contiguous FS available
                spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp, self.j * 2] = (num_slots - 5.5) / 3.5  # number of FSs necessary

            idx, values, lengths = CustomDeepRMSAEnv.rle(available_slots)

            spectrum_obs[idp, self.j * 2 + 1] = (2 * (np.sum(available_slots) - 0.5 * self.num_spectrum_resources) / self.num_spectrum_resources)  # total number available FSs
            av_indices = np.argwhere(values == 1)  # getting indices which have value 1
            if av_indices.shape[0] > 0:
                spectrum_obs[idp, self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4  # avg. number of FS blocks available

        print("Current Service Observation: ", src_node, " -> ", dst_node)

        # adjacency_matrix = np.random.randint(2, size=(self.max_num_nodes, self.max_num_nodes)).reshape(1, self.max_num_nodes * self.max_num_nodes)
        # node_features = np.random.randint(2, size=(self.max_num_nodes, self.k_paths)).reshape(1, self.max_num_nodes * self.k_paths)
        # traffic_src = np.random.randint(2, size=(self.max_num_nodes, )).reshape(1, self.max_num_nodes)
        # traffic_dst = np.random.randint(2, size=(self.max_num_nodes, )).reshape(1, self.max_num_nodes)
        # bit_rate = np.random.uniform(low=-1e6, high=1e6, size=(1,)).reshape(1, 1)
        # spectrum_details = np.random.uniform(low=-1e6, high=1e6, size=((2 * self.j + 3) * self.k_paths,)).reshape(1, (2 * self.j + 3) * self.k_paths)

        adjacency_matrix = k_paths_adjacency_matrix.reshape(1, self.max_num_nodes * self.max_num_nodes)
        node_features = node_features_matrix.reshape(1, self.max_num_nodes * self.k_paths)
        traffic_src = traffic_src_obs.reshape(1, self.max_num_nodes)
        traffic_dst = traffic_dst_obs.reshape(1, self.max_num_nodes)
        bit_rate = bit_rate_obs.reshape(1, 1)
        spectrum_details = spectrum_obs.reshape(1, (2 * self.j + 3) * self.k_paths)

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

    def reset(self, only_episode_counters=False):
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
