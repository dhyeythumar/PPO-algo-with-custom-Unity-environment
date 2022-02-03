from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from pathlib import Path
import numpy as np
from typing import Deque, Dict, List, Tuple
from keras.models import load_model

# import keras.backend as K
# import tensorflow as tf


# Name of the Unity environment binary to be launched
ENV_NAME = "./rl_env_binary/Windows_build/Learning-Agents--r1"
RUN_ID = "train-1"


class Test_FindflagAgent:
    def __init__(self, env: UnityEnvironment):
        MODEL_NAME = self.get_model_name()
        self.env = env
        self.env.reset()  # without this env won't work
        self.behavior_name = self.env.get_behavior_names()[0]
        self.behavior_spec = self.env.get_behavior_spec(self.behavior_name)
        self.state_dims = np.sum(self.behavior_spec.observation_shapes)
        self.n_actions = self.behavior_spec.action_size

        self.actor = load_model(
            MODEL_NAME, custom_objects={"loss": "categorical_hinge"}
        )

    def get_model_name(self) -> str:
        """Get the latest saved actor model name."""
        _dir = "./training_data/model/" + RUN_ID
        basepath = Path(_dir)
        files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())

        for item in files_in_basepath:
            if item.name.find("actor") != -1:
                name = _dir + "/" + item.name
        print("-" * 100)
        print("\t\tUsing {} saved model for testing.".format(name))
        print("-" * 100)
        return name

    def check_done(self, step_result) -> bool:
        """Return the done status for env reset."""
        if len(step_result[1]) != 0:
            return True
        else:
            return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """
        Apply the actions to the env, step the env and return new set of experience.

        Return the next_obs, reward and done response of the env.
        """
        self.env.set_actions(self.behavior_name, action)
        self.env.step()
        step_result = self.env.get_steps(self.behavior_name)
        done = self.check_done(step_result)
        next_obs = np.array([])  # store next observations

        if not done:
            for obs in step_result[0].obs:
                next_obs = np.append(next_obs, obs)  # shape(54,)
            reward = step_result[0].reward[0]
        else:
            for obs in step_result[1].obs:
                next_obs = np.append(next_obs, obs)
            reward = step_result[1].reward[0]
        return next_obs, reward, done

    def get_action(self, action_probs: np.ndarray) -> np.ndarray:
        """Get actions from action probablities."""
        n_agents = 1  # only 1 agent is used in the env.

        action = action_probs[0]
        action = np.clip(action, -1, 1)  # just for confirmation
        return np.reshape(action, (n_agents, self.n_actions))

    def test(self) -> None:
        """Test the trained Actor model."""
        self.env.reset()
        score = 0
        step_result = self.env.get_steps(self.behavior_name)
        observation = np.array([])
        for obs in step_result[0].obs:
            observation = np.append(observation, obs)

        try:
            while True:
                observation = np.expand_dims(observation, axis=0)  # shape(1, 54)
                action_probs = self.actor.predict(observation, steps=1)  # (1, 2)
                action = self.get_action(action_probs)
                next_obs, reward, done = self.step(action)
                observation = next_obs
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
            print("-" * 100)
            print("\t\tException has occured !!\tTesting was interrupted.")
            print("-" * 100)
        self.env.close()


if __name__ == "__main__":
    engine_config_channel = EngineConfigurationChannel()
    engine_config_channel.set_configuration_parameters(
        width=1800, height=900, time_scale=1.0
    )

    env = UnityEnvironment(
        file_name=ENV_NAME, seed=0, side_channels=[engine_config_channel]
    )

    agent = Test_FindflagAgent(env)
    agent.test()
