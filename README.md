# PPO Algorithm with a custom environment

<h4 align="center">
    This repo contains the implementation of the Proximal Policy Optimization algorithm using the Keras library on a custom environment made with Unity 3D engine.
</h4>
</br>

> **Important details about this repository:**
> - Unity engine version used to build the environment = 2019.3.15f1
> - ML-Agents branch = release_1
> - Environment name = Learning-Agents--r1.exe (env. provided in this repo)

**This environment can be used on Windows. I will soon upload the binary files for the Linux system.**


## What’s In This Document
- [Introduction](#introduction)
- [Environment Specific Details](#environment-specific-details)
- [Setup Instructions](#setup-instructions)
- [Getting Started](#getting-started)
- [Motivation & Learnings](#motivation&-learnings)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Introduction
- Check out [**this video**](https://youtu.be/4vwZNTagHsQ) to see the trained agent showing the learned navigation skills to find the flag in a closed environment, which is divided into nine different segments.
- And if you want to see the training phase of this agent, then check out [**this video**](https://youtu.be/eIp36b5lBVM)


## Environment Specific Details
These are some details which you should know before hand. And I think without knowing this, you might get confused because some of the Keras implementations are environment-dependent.

- Observation/State space: Vectorized
- Action space: Continuous [shape(1, 2)]
- Reward System: 
    - -1.0/MaxStep per step (MaxStep is used to reset the env irrespective of goal state) & the same reward is used if the agent crashes into the walls.
    <br>
    - +2 is the agent reaches the goal state.


## Setup Instructions
Install the ML-Agents github repo [release_1_branch](https://github.com/Unity-Technologies/ml-agents/tree/release_1_branch), but if you want to use the different branch version then modify the python APIs to interact with the environment.

- Clone this repos:
```bash
$ git clone --branch release_1 https://github.com/Unity-Technologies/ml-agents.git

$ git clone https://github.com/Dhyeythumar/PPO-algo-with-custom-Unity-environment.git
```

- Create and activate the python virtual environment:
```bash
$ python -m venv myvenv
$ myvenv\Scripts\activate
```

- Install the requirements:
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


## Motivation & Learnings
[**This video**](https://youtu.be/kopoLzvh5jY) by [**OpenAI**](https://openai.com/) inspired me to develop something in the field of reinforcement learning. So for the first phase, I decided to create a simple RL agent who can learn navigation skills. 

Now the first phase is completed I learned many new things and got some of my following questions answered:
- How to create custom 3D environments using the Unity engine?
- How to use ML-Agents (Unity's toolkit for reinforcement learning) to train the RL agents?
- And finally, I learned to implement the PPO algorithm using the Keras library.

**What next?**
So I have started working on the next phase of this project, which will include a multi-agent environment and, I am also planning to increase the difficulty level. So for more updates, stay tuned for the next video on [**youtube**](https://www.youtube.com/channel/UCpKizIKSk8ga_LCI3e3GUig).

## License
Licensed under the [MIT License](./LICENSE).


## Acknowledgements
1. [Unity ML-Agents Python Low Level API](https://github.com/Unity-Technologies/ml-agents/blob/release_1_branch/docs/Python-API.md)
2. [rl-bot-football](https://github.com/ChintanTrivedi/rl-bot-football)