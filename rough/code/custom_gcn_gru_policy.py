from stable_baselines3.common.policies import ActorCriticPolicy


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_class, net_arch=None, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                           features_extractor_class=features_extractor_class,
                                           net_arch=net_arch, **kwargs)