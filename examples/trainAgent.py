import gym
import numpy as np
from gym import spaces
from collections import deque

import cv2
import random
import time
import glob
  
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai_wrapper import AnimalWrapper, AnimalStack, AnimalSkip

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.gail import ExpertDataset
from custom_network_policy_final import LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


worker_id = random.randint(1, 100)
dataset_path = 'dataset/'
tensorboard_log_path = 'tensorboard_log/'
VISUAL_FRAMES_COUNT = 2
VEL_FRAMES_COUNT = 2
SKIP_FRAMES = 1
USE_GREYSCALE_OBSES = False


def create_env_fn(num_actors=1, inference = True, config=None, seed=None):
    env_path = '../env/AnimalAI'
    #worker_id = random.randint(1, 60000)
    # base arena (If nothing is defined!!)
    arena_config_in = ArenaConfig('configs/1-Food.yaml')

    if config is None:
        config = arena_config_in
    else: 
        config = ArenaConfig(config)
    if seed is None:
        seed = 0#random.randint(0, 100500)

    def env():
        worker_id = random.randint(1, 60000)
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
        
    return env

# Define environments
env = create_env_fn(num_actors = 1, inference=False, seed=0)
env = make_vec_env(env, n_envs=4)

# # register policy
register_policy('MyPolicy', LstmPolicy)

# # define algorithm
model = PPO2('MyPolicy', env, n_steps=256)

#########################
# Dataset concatenation #
#########################

def dataset_concatenation(dataset_path):
    '''
    Use only when you have datasets of seperate environments.
    If not, and the code already has a concatenated all_data.npz, ***do not use the function***

    Input: Directory where expert trajectory per environment .npz files are present
    Output: A all_data.npz in the same directory
    '''
    all_npzs = sorted(glob.glob(dataset_path+'*.npz'))
    print(all_npzs)

    npz_path = all_npzs[0]
    data = np.load(npz_path)
    all_data = {'observations': data['observations'],
                'rewards': data['rewards'] ,
                'episode_returns': data['episode_returns'] ,
                'actions': data['actions'],
                'episode_starts': data['episode_starts'] }

    print(all_data['observations'].shape, all_data['rewards'].shape, all_data['episode_returns'].shape, all_data['actions'].shape, all_data['episode_starts'].shape)

    for npz_path in all_npzs[1:]:
        data = np.load(npz_path)
        print(npz_path)
        #print(data['observations'].shape, data['rewards'].shape, data['episode_returns'].shape, data['actions'].shape, data['episode_starts'].shape)
        all_data['observations'] = np.concatenate((all_data['observations'], data['observations']))
        all_data['rewards'] = np.concatenate((all_data['rewards'], data['rewards']))
        all_data['episode_returns'] = np.concatenate((all_data['episode_returns'], data['episode_returns']))
        all_data['actions'] = np.concatenate((all_data['actions'], data['actions']))
        all_data['episode_starts'] = np.concatenate((all_data['episode_starts'], data['episode_starts']))
        print(all_data['observations'].shape, all_data['rewards'].shape, all_data['episode_returns'].shape, all_data['actions'].shape, all_data['episode_starts'].shape)

    all_data['actions'] = all_data['actions'].reshape(-1,1)
    print(all_data['observations'].shape, all_data['rewards'].shape, all_data['episode_returns'].shape, all_data['actions'].shape, all_data['episode_starts'].shape)
    save_path = dataset_path+'all_data'
    np.savez(save_path, **all_data)

#dataset_concatenation(dataset_path)

##################
# Pretrain model #
##################
dataset = ExpertDataset(expert_path=dataset_path+'all_data.npz',
                        traj_limitation=-1, batch_size=256, LSTM=True, envs_per_batch=1)
model.pretrain(dataset, n_epochs=100)

model.save('ppo_model_after_bc')


# ###############
# # Train model #
# ###############

del model
env.close()

# Curriculum implemented is extremely basic one. (Need to improvise later on)
# Curriculum is a newly made folder. In google drive, read the note
all_arenas = sorted(glob.glob('configs/Curriculum/*.yaml'))
print(all_arenas)

model_name = 'ppo_model_after_bc'
all_frames_vec = [200000, 200000, 100000, 200000, 200000, 200000, 200000, 200000] #no of tsteps per curriculum

for i in range(len(all_arenas)):
    # create arena 
    env = create_env_fn(num_actors = 1, inference=False, config=all_arenas[i], seed=0)
    env = make_vec_env(env, n_envs=4)
    print('####################')
    print("##  Curriculum {} ##".format(i))
    print('####################')

    model = PPO2.load(model_name, env, verbose=1, tensorboard_log=tensorboard_log_path, full_tensorboard_log=False)

    frames_idx = all_frames_vec[i]
    print('{} arena is used for training for {} timesteps'.format(all_arenas[i], frames_idx))
    model.learn(total_timesteps=frames_idx)
    model_name = "ppo_model_after_training_arena_{}".format(i)
    model.save(model_name)
    env.close()

    del model
    del env

print('Training complete!!')