import gym
from statistics import Memory
import numpy as np
from typing import Deque, Dict, List, Tuple
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


# Name of the Unity environment binary to be launched
ENV_NAME = "LunarLanderContinuous-v2"
RUN_ID = "GYM_train-1"
RESUME = True  # resume the training process.

# used in Actor-Critic neural network.
NUM_LAYERS = 2
HIDDEN_UNITS = 128
LEARNING_RATE = 3.0e-4

# training parameters
BATCH_SIZE = 128  # (used in .fit())
BUFFER_SIZE = 128  # (experience buffer)
MAX_STEPS = 50  #
NUM_EPOCH = 3  #
SUMMARY_FREQ = 10  # (granularity of graphs in TensorBoard)

# used for advantage cal. in GAE function.
LAMBDA = 0.95  # tau
GAMMA = 0.99
# used in ppo loss function.
EPSILON = 0.2  # (LOSS_CLIPPING)
CRITIC_DISCOUNT = 0.5
BETA = 0.001  # (ENTROPY_BETA)    5e-3

# I don't know what to do with this! 😅 [but its used in ML-Agents trainer_config.yaml file]
# I think its size of mini_batch used for .predict_on_batch() method.
TIME_HORIZON = 128

# actor_loss = sum(actor_losses) / len(actor_losses)
# critic_loss = sum(critic_losses) / len(critic_losses)
# total_loss = sum(total_losses) / len(total_losses)


class Actor_Critic:  # keras functional model
    # TODO :: Its this giving 0's only?
    @staticmethod
    def ppo_loss(oldpolicy_probs, advantage, reward, value):
        def loss(y_true, y_pred):
            newpolicy_probs = y_true * y_pred
            old_prob = y_true * oldpolicy_probs
            # print("newpolicy_probs: ", K.get_value(newpolicy_probs))

            ratio = newpolicy_probs / (old_prob + 1e-10)
            clip_ratio = K.clip(ratio, min_value=1 - EPSILON, max_value=1 + EPSILON)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage

            # single values
            actor_loss = -K.mean(K.minimum(surrogate1, surrogate2))
            critic_loss = K.mean(K.square(reward - value))
            entropy_loss = K.mean(
                -(newpolicy_probs * K.log(K.abs(newpolicy_probs) + 1e-10))
            )
            actor_loss_ = actor_loss - tf.constant(BETA) * entropy_loss
            total_loss = tf.constant(CRITIC_DISCOUNT) * critic_loss + actor_loss_
            return actor_loss_

        return loss

    @staticmethod
    def actor_model(input_dims, output_dims):
        """Keras implementation for Actor network."""
        observation = Input(shape=(input_dims,), name="observation_input")
        oldpolicy_probs = Input(shape=(output_dims,), name="old_prediction_input")
        advantage = Input(shape=(1,), name="advantage_input")
        reward = Input(shape=(1,), name="reward_input")
        value = Input(shape=(1,), name="value_input")

        x = Dense(HIDDEN_UNITS, activation="tanh", name="fc1")(observation)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_UNITS, activation="tanh")(x)
        policy = Dense(output_dims, activation="tanh", name="policy")(x)

        actor_network = Model(
            inputs=[observation, oldpolicy_probs, advantage, reward, value],
            outputs=[policy],
        )
        actor_network.compile(
            optimizer=Adam(lr=LEARNING_RATE),
            loss=Actor_Critic.ppo_loss(
                oldpolicy_probs=oldpolicy_probs,
                advantage=advantage,
                reward=reward,
                value=value,
            ),
            run_eagerly=True,
        )
        actor_network.summary()
        return actor_network

    @staticmethod
    def critic_model(input_dims):
        """Keras implementation for Critic network."""
        observation = Input(shape=(input_dims,), name="observation_input")

        x = Dense(HIDDEN_UNITS, activation="tanh", name="fc1")(observation)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_UNITS, activation="tanh")(x)
        V = Dense(1, name="values")(x)  # activation='tanh'

        critic_network = Model(inputs=[observation], outputs=[V])
        critic_network.compile(optimizer=Adam(lr=LEARNING_RATE), loss="mse")
        critic_network.summary()
        return critic_network


class FindflagAgent:
    def __init__(self, env: gym.Env):
        self.base_model_dir = "training_data/model/" + RUN_ID

        self.memory = Memory(RUN_ID)  # for tensorboard summaries
        self.env = env

        self.state_dims = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.dummy_n = np.zeros((1, self.n_actions))
        self.dummy_1 = np.zeros((1, 1))

        self.actor = Actor_Critic.actor_model(
            input_dims=self.state_dims, output_dims=self.n_actions
        )
        self.critic = Actor_Critic.critic_model(input_dims=self.state_dims)

    def save_model_weights(self, steps: int) -> None:
        """Save the Actor-Critic network weights."""
        actor_path = self.base_model_dir + "/checkpoints/actor_weights_{}.ckpt"
        critic_path = self.base_model_dir + "/checkpoints/critic_weights_{}.ckpt"

        self.actor.save_weights(actor_path.format(steps))
        self.critic.save_weights(critic_path.format(steps))

    def load_model_weights(self):
        """Load the stored Actor-Critic network weights (if avaliable)."""
        _dir = self.base_model_dir + "/checkpoints/"
        latest = tf.train.latest_checkpoint(_dir)

        if latest == None:
            print("-" * 100)
            print(
                "[INFO]\tNO saved model to resume the training. Starting with new traning process."
            )
            print("-" * 100)
            return 0
        else:
            print("-" * 100)
            print("[INFO]\tLoading the saved model & resume the training process")
            print("-" * 100)

        self.actor.load_weights(latest.replace("critic", "actor"))
        self.critic.load_weights(latest)

        # return last training step number.
        return int(latest.split("_")[-1].split(".")[0])

    def save_model(self, steps: int):
        """Save the Actor-Critic model."""
        actor_path = self.base_model_dir + "/actor_{}.hdf5"
        critic_path = self.base_model_dir + "/critic_{}.hdf5"

        self.actor.save(actor_path.format(steps))
        self.critic.save(critic_path.format(steps))

    def get_advantages(self, values, masks, rewards):
        """Cal discounted reward & GAE values for the collected experiences."""
        dis_returns = []  # discounted_returns
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i + 1] * masks[i] - values[i]
            gae = delta + GAMMA * LAMBDA * masks[i] * gae
            dis_returns.insert(0, gae + values[i])

        adv = np.array(dis_returns) - values[:-1]
        return dis_returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def get_action(self, action_probs: np.ndarray, train: bool):
        """Get actions from action probablities."""
        if train is True:
            action = action_probs[0] + np.random.normal(
                loc=0, scale=1.0, size=action_probs[0].shape
            )  # shape(2,)
        else:
            action = action_probs[0]

        action = np.clip(action, -1, 1)
        return np.reshape(action, (self.n_actions,)), action.copy()

    def get_experience(self):
        """
        Collect the agent experiences in a buffer by apply the policy network for training process.

        Returns a dictionary of states, actions, old_predictions, rewards, values, masks, & episode length.
        """
        states = []
        actions = []
        old_predictions = []
        rewards = []
        values = []
        masks = []
        episode_lens = []
        counter = 0

        state = self.env.reset()

        while len(states) < BUFFER_SIZE:
            state = K.expand_dims(state, 0)
            action_probs = self.actor.predict(
                [state, self.dummy_n, self.dummy_1, self.dummy_1, self.dummy_1], steps=1
            )  # shape(1, 2)
            q_value = self.critic.predict([state], steps=1)

            action, action_matrix = self.get_action(action_probs, True)
            # print("action_probs", action_probs)
            # print("action", action)
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            mask = not done

            states.append(state)
            actions.append(action_matrix)
            old_predictions.append(action_probs)
            # used in GAE
            rewards.append(reward)
            values.append(q_value)
            masks.append(mask)

            counter += 1
            if done:
                episode_lens.append(counter)
                counter = 0
                next_state = self.env.reset()
            state = next_state

        if len(episode_lens) == 0:
            episode_lens.append(0)

        state = K.expand_dims(state, 0)
        q_value = self.critic.predict(state, steps=1)
        values.append(q_value)
        discounted_returns, advantages = self.get_advantages(values, masks, rewards)

        # if self.hyper_params.standardize_advantage:
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # reshaping
        states = np.reshape(states, (BUFFER_SIZE, self.state_dims))
        actions = np.reshape(actions, (BUFFER_SIZE, self.n_actions))
        old_predictions = np.reshape(old_predictions, (BUFFER_SIZE, self.n_actions))

        rewards = np.reshape(rewards, (BUFFER_SIZE, 1))
        values = np.reshape(values, (len(values), 1))
        advantages = np.reshape(advantages, (BUFFER_SIZE, 1))
        discounted_returns = np.reshape(discounted_returns, (BUFFER_SIZE, 1))

        return {
            "states": states,
            "actions": actions,
            "old_predictions": old_predictions,
            "rewards": rewards,
            "values": values[:-1],
            "advantages": advantages,
            "discounted_returns": discounted_returns,
            "episode_lens": episode_lens,
        }

    def train(self) -> None:
        """Train the Actor & Critic network on collected experiences stored in buffer."""
        if RESUME == True:
            start_pt = self.load_model_weights()
        step = 1 if (start_pt == 0) else start_pt + 1

        try:
            while step <= MAX_STEPS:
                buffer = self.get_experience()

                states = buffer["states"]
                actions = buffer["actions"]
                old_predictions = buffer["old_predictions"]
                rewards = buffer["rewards"]
                values = buffer["values"]
                advantages = buffer["advantages"]
                discounted_returns = buffer["discounted_returns"]
                episode_lens = buffer["episode_lens"]

                actor_loss = self.actor.fit(
                    [states, old_predictions, advantages, rewards, values],
                    [actions],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    epochs=NUM_EPOCH,
                    verbose=0,
                )
                critic_loss = self.critic.fit(
                    [states],
                    [discounted_returns],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    epochs=NUM_EPOCH,
                    verbose=0,
                )

                # TODO :: check actor_loss
                print(actor_loss.history["loss"], critic_loss.history["loss"])

                # storing data having length = SUMMARY_FREQ & then find the means.
                self.memory.add_data(
                    np.mean(rewards),  # try with max
                    np.max(episode_lens),
                    np.mean(actor_loss.history["loss"]),
                    np.mean(critic_loss.history["loss"]),
                    np.mean(advantages),
                    dict(self.actor.optimizer.get_config())["learning_rate"],
                    dict(self.critic.optimizer.get_config())["learning_rate"],
                )

                if step % SUMMARY_FREQ == 0:
                    self.memory.terminal_logs(step)
                    self.memory.tensorboard_logs(step)
                    # save checkpoint logs
                    self.save_model_weights(step)
                step += 1
        except KeyboardInterrupt as ex:
            print("-" * 100)
            print("Exception has occured !!")
            print("Learning was interrupted. Please wait while the model is saved.")
            print("-" * 100)
            self.memory.terminal_logs(step)
            self.memory.clear_memory()
            # save checkpoint logs
            self.save_model_weights(step)
        finally:
            self.save_model(step)
            self.env.close()


if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    agent = FindflagAgent(env)
    agent.train()
