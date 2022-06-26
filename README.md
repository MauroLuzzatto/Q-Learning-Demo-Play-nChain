# Q-Learning Notebook - Play the N-Chain Environment with three Agents
This repository contains a Jupyter Notebook with an implementation of a `Q-Learning` agent, which learns to solve the n-Chain `OpenAI Gym` environment 

This notebook is inspired by the following notebook: 
[Deep Reinforcement Learning Course Notebook](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb)

## Q-Learning

The notebook contains a `Q-Learning` algorithm implementation and a training loop to solve the n-Chain OpenAI Gym environment. The `Q-Learning` algorithm is an oﬀ-policy temporal-difference control algorithm [1]:


<img src="/images/Sutton_Barto.png" alt="Q-Learning" width="600"/>

[Image](http://incompleteideas.net/book/the-book-2nd.html) taken from **Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, Second edition, 2014/2015, page 158**

## The Q-Learning Agents

In this notebook we let different q-learning agents play the N-Chain evironment and see how they perform in the game. The following agents are implemented:

- 🤓 Smart Agent 1: the agent explores and takes future rewards into account
- 🤑 Greedy Agent 2: the agent cares only about immediate rewards (small gamma)
- 😳 Shy Agent 3: the agent doesn't explore the environment (small epsilon)



## The n-Chain Environment

The n-Chain environment is taken from the `OpenAI Gym` module. Documentation: 

[n-Chain environment](https://gym.openai.com/envs/NChain-v0/)

The image below shows an example of a 5-Chain (n = 5) environment with 5 states. `a` stands for action and `r` for the reward ([Image Source](https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)).
<!-- ![NChain](images/NChain-illustration.png)
 -->
<img src="/images/NChain-illustration.png" alt="NChain" width="600"/>

### States

This environment contains a chain with n positions, and every chain position corresponds to a possible state the agent can be in:


|  state    |  description|
|---        |--- |
| n (default n=5)    | n-th postion on the chain |


### Actions and Rewards

The agent can move along the chain using two actions for which the agent will get a different rewards:

|action   | reward  | description   |  
|---|---|---|
|  0 | get no reward  |   move forward along the chain (state = n+1) | 
|  1 |   get a small reward of 2 | jump back to state 0  |  


**The end of the chain presents a large reward of 10, and while standing at the end of the chain and still moving forward (action 0), the large reward can be gained repeatedly**.



## Additional Resources About Reinforcement Learning
- [OpenAI Gym](https://gym.openai.com/): Gym is a toolkit for developing and comparing reinforcement learning algorithms from OpenAI
- [OpenAI Baselines](https://github.com/openai/baselines): OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms
- [Spining Up AI](https://spinningup.openai.com): This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning
- [A Long Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html): Great blog post from Lilian Weng, where she is briefly going over the field of Reinforcement Learning (RL), from fundamental concepts to classic algorithms
- [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html): Another great blog post from Lilian Weng, where she writes about policy gradient algorithms



