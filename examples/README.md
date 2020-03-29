# Animal-AI Imitation Learning and Curriculum Learning on a PPO agent

## Introduction

Fill up the basic documentation here. (Some images, where we started and all (keep it short)) 

## **Our Proposed Solution**

We implemented a reformed architecture which uses Proximal Policy Optimization for training the agent and also uses Behavior Cloning for incorporating Expert Trajectories, which considerably reduce the training time and also improve agent’s performance on some of the harder tasks compared to an agent without Imitation Learning techniques.

The Model Architecture is as shown below:

![Model Arch](https://github.com/addy369/AnimalAI/blob/master/Images/ModelArch.JPG)

We also used Behavior Cloning, whose basic structure is as follows:
*Courtesy of Caude Sammut, Encyclopedia of Machine Learning*

![BC](https://github.com/addy369/AnimalAI/blob/master/Images/BC.gif)

### Results

![Image](https://github.com/addy369/AnimalAI/blob/master/Images/RLImage.JPG)

## Code Structure:
* `trainAgent.py`
  * Main code to run
* `animalai_wrapper.py`
  * Environment wrapper designed by the last winner of the competition.
  * Future teams needs to work on the wrapper too. **Future Work**
* `gym_loop.py` and `unity_loop.py`
  * Examples of how to use a gym and unity environment in code
* `custom_network_policy_final.py`
  * Consists of our model. (Inspired by the winner of the competition)
  * Currently does not include velocity components in the entire pipeline. **Future Work**
* `configs/Curriculum`
  * The curriculum and its arenas.
  * Read the **note** to understand the arenas.
  * Curriculum implelemted is extremely basic. May have issues of forgetting earlier training. **Future Work**
* `dataset`
  * The expert trajectories collected for various arenas.[Link](./dataset/README.md)
  * Need to improve imitation learning performance and come up with a metric to represent the improvement **Future Work**
* Other codefiles:
  * Read AnimalAI package doc to understand those. Not used in our codebase.

## How to use the code:

1. Run `python trainAgent.py`
  - By default, the code runs behavior cloning and then starts with training curriculum.
  - After BC and every change of arena in curriculum, the model is saved.
  - The tensorboard logs are stored at `\tmp`. To change the path, enter ppo2.py in stablebaselines. 
  - Logs are space consuming, so try saving it someplace with harddrive space.
  
2. Testing of the model:
  - Follow animalai package documentation.
