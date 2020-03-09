from animalai.envs import UnityEnvironment
import trainMLAgents.py 

env= init_environment() # environment is created

#instantiate the agent??
tc = TrainerController(model_path, summaries_dir, run_id + '-' + str(sub_id),
                       save_freq, maybe_meta_curriculum,
                       load_model, train_model,
                       keep_checkpoints, lesson, external_brains, run_seed, arena_config_in)
tc.start_learning(env, trainer_config)


info = env.step(vector_action=take_action_vector)



brain = info['Learner']

brain.visual_observations   # list of 4 pixel observations, each of size (84x84x3)
brain.vector_observation    # list of 4 speeds, each of size 3
brain.reward                # list of 4 float rewards
brain.local_done            # list of 4 booleans to flag if each agent is done or not
env.reset(arenas_configurations=arena_config,     # A new ArenaConfig to use for reset, leave empty to use the last one provided
        train_mode=True                           # True for training
        )
env.close()