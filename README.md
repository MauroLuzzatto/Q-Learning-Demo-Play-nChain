# Reinforcement-Learning--Notebook
### Introduction to Reinforcement Learning and the Q-Learning Algorithm

This notebook is supposed to give someone interested in reinforcement learning a short and easy understandable introduction. Additionally, the notebook shows how a Q-Learning algorithm can be implemented and used within an OpenAI Gym environment.


This notebook is based on the following Q-learning algorithm code:
- https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb

The n-Chain environment is taken from the OpenAI Gym module:
- Official documentation: https://gym.openai.com/envs/NChain-v0/

### The n-Chain Environment

This environment contains of a linear chain of states, where the agent moving in this environment can take two actions (for which the agent will get a different reward):
- action 0 = move forward along the chain, but get no reward
- action 1 = move backward to state 0, get small reward of 2

The end of the chain, however, presents a large reward of 10, and standing at the end of the chain and still moving 'forward' the large reward can be gained repeatedly.

![NChain](/NChain-illustration.png)
(Image taken from: https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)
