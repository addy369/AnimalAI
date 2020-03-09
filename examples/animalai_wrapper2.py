import gym
import numpy as np
from gym import spaces
from collections import deque


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # modify obs
        obs=obs.reshape(84,84,1)
        return obs