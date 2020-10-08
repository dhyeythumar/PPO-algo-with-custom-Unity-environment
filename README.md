# PPO Algorithm with a custom environment

<h4 align="center">
    This repo contains the implementation of the Proximal Policy Optimization algorithm using the Keras library on a custom environment made with Unity 3D engine.
</h4>
</br>

> **Important details about this repository:**
> - Unity engine version used to build the environment = [2019.3.15f1](https://unity3d.com/get-unity/download/archive)
> - ML-Agents branch = [release_1](https://github.com/Unity-Technologies/ml-agents/tree/release_1_branch)
> - Environment binary:
>      - For Windows = [Learning-Agents--r1 (.exe)](https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment/tree/main/rl_env_binary/Windows_build)
>      - For Linux(Headless/Server build) = [RL-agent (.x86_64)](https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment/tree/main/rl_env_binary/Linux_headless_build)
>      - For Linux(Normal build) = [RL-agent (.x86_64)](https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment/tree/main/rl_env_binary/Linux_build)

**Windows environment binary is used in this repo. But if you want to use the Linux environment binary, then change the ENV_NAME in train.py & test.py scripts to the correct path pointing to those binaries stored [over here](https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment/tree/main/rl_env_binary).**


## What’s In This Document
- [Introduction](#introduction)
- [Environment Specific Details](#environment-specific-details)
- [Setup Instructions](#setup-instructions)
- [Getting Started](#getting-started)
- [Motivation and Learning](#motivation-and-learning)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Introduction
- Check out [**this video**](https://youtu.be/4vwZNTagHsQ) to see the trained agent using the learned navigation skills to find the flag in a closed environment, which is divided into nine different segments.
- And if you want to see the training phase/process of this agent, then check out [**this video**](https://youtu.be/eIp36b5lBVM).


## Environment Specific Details
These are some details which you should know before hand. And I think without knowing this, you might get confused because some of the Keras implementations are environment-dependent.

[**Check this doc for detailed information.**](./Environment_Details.md)

A small overview of the environment:
- Observation/State space: Vectorized (unlike Image)
- Action space: Continuous  (unlike Discrete)
- Action shape: (num of agents, 2)   (Here num of agents alive at every env step is 1, so shape(1, 2))
- Reward System: 
    - (1.0/MaxStep) per step (MaxStep is used to reset the env irrespective of achieving the goal state) & the same reward is used if the agent crashes into the walls.
    - +2 if the agent reaches the goal state.


## Setup Instructions
Install the ML-Agents github repo [release_1 branch](https://github.com/Unity-Technologies/ml-agents/tree/release_1_branch), but if you want to use the different branch version then modify the python APIs to interact with the environment.

Clone this repos:
```bash
$ git clone --branch release_1 https://github.com/Unity-Technologies/ml-agents.git

$ git clone https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment.git
```

Create and activate the python virtual environment:
```bash
$ python -m venv myvenv
$ myvenv\Scripts\activate
```

Install the dependencies:
```bash
$ pip install -e ./ml-agents/ml-agents-envs
$ pip install tensorflow
$ pip install keras
$ pip install tensorboardX
```


## Getting Started
Now to start the training process use the following commands:
```bash
$ cd PPO-algo-with-custom-Unity-environment
$ python train.py
```

Activate the tensorboard:
```bash
$ tensorboard --logdir=./training_data/summaries --port 6006
```


## Motivation and Learning 
[**This video**](https://youtu.be/kopoLzvh5jY) by [**OpenAI**](https://openai.com/) inspired me to develop something in the field of reinforcement learning. So for the first phase, I decided to create a simple RL agent who can learn navigation skills. 

After completing the first phase, I gained much deeper knowledge in the RL domain and got some of my following questions answered:
- How to create custom 3D environments using the Unity engine?
- How to use ML-Agents (Unity's toolkit for reinforcement learning) to train the RL agents?
- And I also learned to implement the PPO algorithm using the Keras library. :smiley:

**What's next?** 🤔

So I have started working on the next phase of this project, which will include a multi-agent environment setup and, I am also planning to increase the difficulty level. So for more updates, stay tuned for the next video on my [**youtube channel**](https://www.youtube.com/channel/UCpKizIKSk8ga_LCI3e3GUig).

## License
Licensed under the [MIT License](./LICENSE).


## Acknowledgements
1. [Unity ML-Agents Python Low Level API](https://github.com/Unity-Technologies/ml-agents/blob/release_1_branch/docs/Python-API.md)
2. [rl-bot-football](https://github.com/ChintanTrivedi/rl-bot-football)
