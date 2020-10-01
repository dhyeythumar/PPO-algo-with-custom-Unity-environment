from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
import numpy as np

# Name of the Unity environment binary to launch
ENV_NAME = "./rl_env_binary/Windows_build/Learning-Agents--r1"

engine_config_channel = EngineConfigurationChannel()
engine_config_channel.set_configuration_parameters(
    width=1800, height=900, time_scale=1.0
)
env = UnityEnvironment(
    file_name=ENV_NAME, seed=1, side_channels=[engine_config_channel]
)
env.reset()  # Reset the environment

# Set the default brain to work with
behavior_name = env.get_behavior_names()[0]
behavior_spec = env.get_behavior_spec(behavior_name)
n_actions = behavior_spec.action_size  # => 2
state_dims = np.sum(behavior_spec.observation_shapes)  # total obs => 54

# --- Env Spec ---
if behavior_spec.is_action_continuous():
    print("Action space is CONTINUOUS i.e {0, 0.1, 0.2}")
else:
    print("Action space is DISCRETE i.e {0, 1, 2}")
    print(behavior_spec.discrete_action_branches)

print("\nbehavior_spec.observation_shapes :: ", end="")
print(behavior_spec.observation_shapes)  # => [(52,), (2,)]

# ----------------------------------------------------------------------
# Get the state/obs of an agent
step_result = env.get_steps(behavior_name)  # shape(2,)

# Examine the state space for the first observation for the first agent
print("\nAgent observation: \n{}\n".format(step_result[0].obs))
# => [shape(1, 52), shape(1, 2)]

# There are 2 obs vectors (Ray cast vals & velocity vals)
for obs in step_result[0].obs:
    print(obs.shape)

print(step_result[0].__dict__)
print(step_result[1].__dict__)  # data filled at the end of episode.

try:
    for episode in range(10):  # running for 10 episodes.
        print("Starting with a new episode...\n\n")
        env.reset()
        step_result = env.get_steps(behavior_name)
        done = False
        episode_rewards = 0
        end_episode_rewards = 0
        # i = 0;
        while not done:  # running for 1 episode i.e 5000 max_steps
            n_agents = len(step_result[0])

            # if behavior_spec.is_action_continuous():
            action = np.random.randn(n_agents, n_actions)
            action = np.clip(action, -1, 1)
            print(action)
            env.set_actions(behavior_name, action)
            env.step()

            step_result = env.get_steps(behavior_name)
            episode_rewards += step_result[0].reward[0]
            end_episode_rewards += (
                step_result[1].reward[0] if len(step_result[1]) else 0
            )
            done = step_result[1].max_step[0] if len(step_result[1]) else False
            # i += 1
        print(
            "\n\nTotal reward in this episode: {} :: {}".format(
                episode_rewards, end_episode_rewards
            )
        )
        # print(i) # will give 1000 as o/p when 5000 max_step is hit (i.e after 5 step a decision is asked).
except (
    KeyboardInterrupt,
    UnityCommunicationException,
    UnityEnvironmentException,
    UnityCommunicatorStoppedException,
) as ex:
    print("-" * 100)
    print("Exception has occured !!")
    print("Testing of env was interrupted.")
    print("-" * 100)
finally:
    print("Closing the env")

env.close()