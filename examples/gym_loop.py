from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment


import random

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


def create_agent_fn(sess, env, summary_writer):
    return rainbow_agent.RainbowAgent(sess=sess, num_actions=env.action_space.n, summary_writer=summary_writer)


# run_experiment.load_gin_configs(gin_files, None)
# runner = run_experiment.Runner(base_dir=base_dir,
#                                create_agent_fn=create_agent_fn,
#                                create_environment_fn=create_env_fn)
# runner.run_experiment()




# def create_animal(num_actors=1, inference = True, config=None, seed=None):
#     from animalai.envs.gym.environment import AnimalAIEnv
#     from animalai.envs.arena_config import ArenaConfig
#     import random
   
#    	BASE_DIR="/home/aditya/AnimalAI-Olympics-master/examples/configs"
#     arena_config_in = ArenaConfig(BASE_DIR + "1-Food.yaml")

#     if config is None:
#         config = arena_config_in
#     else: 
#         config = ArenaConfig(config)
#     if seed is None:
#         seed = 0#random.randint(0, 100500)
        
#     env = AnimalAIEnv(environment_filename=env_path,
#                       worker_id=worker_id,
#                       n_arenas=num_actors,
#                       seed = seed,
#                       arenas_configurations=config,
#                       greyscale = False,
#                       docker_training=False,
#                       inference = inference,
#                       retro=False,
#                       resolution=84
#                       )
#     env = AnimalSkip(env, skip=SKIP_FRAMES)                  
#     env = AnimalWrapper(env)
#     env = AnimalStack(env,VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, greyscale=USE_GREYSCALE_OBSES)
#     return env


# configurations = {
#     'AnimalAI' : {
#         'ENV_CREATOR' : lambda : create_animal(),
#         'VECENV_TYPE' : 'ANIMAL'
#     },
#     'AnimalAIRay' : {
#         'ENV_CREATOR' : lambda inference=False, config=None: create_animal(1, inference, config=config),
#         'VECENV_TYPE' : 'RAY'
#     },

# }


# def get_obs_and_action_spaces(name):
#     env = configurations[name]['ENV_CREATOR']()
#     observation_space = env.observation_space
#     action_space = env.action_space
#     env.close()
#     return observation_space, action_space

# def register(name, config):
#     configurations[name] = config


# Create environment
env = create_env_fn()

# Instantiate the agent
agent=create_agent_fn()

# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
