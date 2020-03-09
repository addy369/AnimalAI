import gym
import numpy as np
from gym import spaces
from collections import deque

import cv2
import hyperparameters
import animalai_wrapper
import animalai_wrapper2
  
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment


from stable_baselines import PPO1
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from custom_network_policy4 import LstmPolicy

import random

from hyperparameters import USE_GREYSCALE_OBSES, VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, SKIP_FRAMES, BASE_DIR

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)
arena_config_in = ArenaConfig('configs/1-Food.yaml')
base_dir = 'models/dopamine'
gin_files = ['configs/rainbow.gin']

def create_env_fn():
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=1,
                      arenas_configurations=arena_config_in,
                      docker_training=False,
                      retro=True)
    return env

env=create_env_fn()
env=animalai_wrapper2.ObservationWrapper(env)
obs = env.reset()
obs, rewards, dones, info = env.step(1)
print(obs.shape,rewards,dones)


register_policy('MyPolicy', LstmPolicy)


model = PPO1('MyPolicy', env,verbose=1)
# Train the agent
model.learn(total_timesteps=1)
# Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# model = DQN.load("dqn_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)

# Enjoy trained agent
obs = env.reset()
for i in range(1):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

