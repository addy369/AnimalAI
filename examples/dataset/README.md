# Dataset for Behavior Cloning

We generated around 50 expert trajectories for various arenas and used those trajectories for pretraining the agent model. 
Link to the dataset: [here](https://drive.google.com/drive/folders/10anPYxoCEPswmeTHQ-C2WArceJOIu5KO) 

In the **trainAgent.py** code, there is a data concatenation function that combines all these seperate environment trajectories dataset into a single `all_data.npz`. By default, it is commented out, but use that function only once for creating `all_data.npz`.
