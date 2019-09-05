# Q-Learning - Demo Notebook

## Getting Started

Run the Jupyter Notebook:
`q_learning_notebook.ipynb`

## Introduction to Reinforcement Learning and the Q-Learning Algorithm

The purpose of this notebook is to give someone interested in Reinforcement Learning (RL) a short and hopefully understandable introduction. The notebook shows a Q-Learning algorithm implementation (a type of RL Algorithm, an "Off-Policy algorithm for Temporal Difference learning") and how this algorithm can be used to solve a task within an OpenAI Gym environment (library of RL environments).

The n-Chain environment is taken from the OpenAI Gym module:
- [n-Chain](https://gym.openai.com/envs/NChain-v0/): Official Documentation

This notebook is inspired by the following notebook:
- [Deep Reinforcement Learning Course Notebook](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb)


## The n-Chain Environment

This environment contains a linear chain of states, where the agent moving in this environment can take two actions (for which the agent will get a different reward):
- action 0 = move forward along the chain, but get no reward
- action 1 = move backward to state 0, get a small reward of 2

The end of the chain, however, presents a large reward of 10, and standing at the end of the chain and still moving 'forward' the large reward can be gained repeatedly.

The image below shows an example of a 5-Chain (n = 5) environment with 5 states. "a" stands for action and "r" for the reward.

![NChain](/NChain-illustration.png)
[Image Source](https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)

## Further Information about Reinforcement Learning
- [OpenAI Gym](https://gym.openai.com/): Gym is a toolkit for developing and comparing reinforcement learning algorithms from OpenAI
- [OpenAI Baselines](https://github.com/openai/baselines): OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms
- [Spining Up AI](https://spinningup.openai.com): This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning
- [A Long Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html): Great blog post from Lilian Weng, where she is briefly going over the field of Reinforcement Learning (RL), from fundamental concepts to classic algorithms
- [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html): Another great blog post from Lilian Weng, where she writes about policy gradient algorithms
