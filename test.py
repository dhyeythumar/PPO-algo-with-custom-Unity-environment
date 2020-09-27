from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from pathlib import Path
import numpy as np
from typing import Deque, Dict, List, Tuple
from keras.models import load_model
import keras.backend as K
import tensorflow as tf


# Name of the Unity environment binary to be launched
ENV_NAME        = "./rl_env_binary/Windows_build/Learning-Agents--r1"
RUN_ID          = "train-1"


class Test_FindflagAgent:

    def __init__(self, env: UnityEnvironment):

        MODEL_NAME = self.get_model_name()
        self.env = env
        self.env.reset()   # without this env won't work
        self.behavior_name = self.env.get_behavior_names()[0]
        self.behavior_spec = self.env.get_behavior_spec(self.behavior_name)
        self.state_dims = self.behavior_spec.observation_shapes[0][0]
        self.n_actions = self.behavior_spec.action_size

        self.actor = load_model(MODEL_NAME, custom_objects={'loss': 'categorical_hinge'})

    def get_model_name(self) -> str:
        _dir = "./training_data/model/" + RUN_ID
        basepath = Path(_dir)
        files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())

        # get the latest actor's saved model file name. 
        for item in files_in_basepath:
            if (item.name.find("actor") != -1):
                name = _dir + "/" + item.name

        print("-"*100)
        print("\t\tUsing {} saved model for testing.".format(name))
        print("-"*100)
        return name

    def check_done(self, step_result) -> bool:
        if len(step_result[1]) != 0:
            return True
        else:
            return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Return the next_state, reward and done response of the env."""
        self.env.set_actions(self.behavior_name, action)
        self.env.step()
        step_result = self.env.get_steps(self.behavior_name)
        done = self.check_done(step_result)
        if not done:
            next_state = step_result[0].obs[0]
            reward = step_result[0].reward[0]
        else:
            next_state = step_result[1].obs[0]
            reward = step_result[1].reward[0]
        return next_state, reward, done

    def get_action(self, state: np.ndarray) -> np.ndarray:

        action_probs = self.actor.predict(state, steps=1)  # (1, 2)
        action = action_probs[0]
        action = np.clip(action, -1, 1)  # just for confirmation
        return np.reshape(action, (1, self.n_actions))

    def test(self) -> None:
        self.env.reset()
        step_result = self.env.get_steps(self.behavior_name)
        state = step_result[0].obs[0]
        score = 0

        try:
            while True:
                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                if done:
                    print("Score :: ", score)
                    score = 0
                    self.env.reset()
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
        ) as ex:
            print("-"*100)
            print("\t\tException has occured !!\tTesting was interrupted.")
            print("-"*100)
        self.env.close()

if __name__ == '__main__':
    engine_config_channel = EngineConfigurationChannel()
    engine_config_channel.set_configuration_parameters(width=1800, height=900, time_scale=1.0)

    env = UnityEnvironment(file_name=ENV_NAME, seed=2, side_channels=[engine_config_channel])

    agent = Test_FindflagAgent(env)
    agent.test()
