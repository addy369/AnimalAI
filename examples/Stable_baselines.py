import gym
import numpy as np
from gym import spaces
from collections import deque

import cv2
import random
  
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai_wrapper import AnimalWrapper, AnimalStack, AnimalSkip

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from custom_network_policy_final import LstmPolicy

worker_id = random.randint(1, 100)
arena_config_in = 'configs/1-Food.yaml'
VISUAL_FRAMES_COUNT = 2
VEL_FRAMES_COUNT = 2
SKIP_FRAMES = 1
USE_GREYSCALE_OBSES = False


def create_env_fn(num_actors=1, inference = True, config=None, seed=None):
    env_path = '../env/AnimalAI'
    worker_id = random.randint(1, 60000)
    # base arena (If nothing is defined!!)
    arena_config_in = ArenaConfig('configs/1-Food.yaml')

    if config is None:
        config = arena_config_in
    else: 
        config = ArenaConfig(config)
    if seed is None:
        seed = 0#random.randint(0, 100500)
        
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=num_actors,
                      seed = seed,
                      arenas_configurations=config,
                      greyscale = False,
                      docker_training=False,
                      inference = inference,
                      retro=False,
                      resolution=84
                      )
    env = AnimalSkip(env, skip=SKIP_FRAMES)                  
    env = AnimalWrapper(env)
    env = AnimalStack(env,VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, greyscale=USE_GREYSCALE_OBSES)
    return env


env=create_env_fn(num_actors = 1, inference=False, config=arena_config_in, seed=0)

register_policy('MyPolicy', LstmPolicy)

model = PPO1('MyPolicy', env,verbose=1)
# Train the agent
model.learn(total_timesteps=1)
# Save the agent
model.save("dqn_lunar")

# Enjoy trained agent
obs = env.reset()
for i in range(1):
    #action, _states = model.predict(obs)
    action = 1
    obs, rewards, dones, info = env.step(action)
    #env.render()
    print(obs[0], obs[1], obs[0].shape, obs[1].shape, rewards, dones)

