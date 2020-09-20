from tensorboardX import SummaryWriter
import numpy as np


class Memory:
    """
    This memory class is used to store the data for Tensorboard summary
    and Terminal logs.
    Length of the data stored is equal to SUMMARY_FREQ used while training.
    Data length = BUFFER_SIZE is crunched to a single value before stored in this class.
    """
    def __init__(self, RUN_ID):

        self.base_tb_dir = "./training_data/summaries/" + RUN_ID
        self.writer = SummaryWriter(self.base_tb_dir)

        # lists to store data length = SUMMARY_FREQ
        self.rewards         = []
        self.episode_lens    = []
        self.actor_losses    = []
        self.critic_losses   = []
        self.advantages      = []
        self.actor_lrs       = []  # actor learning rate
        self.critic_lrs      = []  # critic learning rate

    def add_data(self, reward, episode_len, actor_loss, critic_loss, advantage, actor_lr, critic_lr):
        self.rewards.append(reward)
        self.episode_lens.append(episode_len)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.advantages.append(advantage)
        self.actor_lrs.append(actor_lr)
        self.critic_lrs.append(critic_lr)

    def clear_memory(self):
        self.rewards.clear()
        self.episode_lens.clear()
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.advantages.clear()
        self.actor_lrs.clear()
        self.critic_lrs.clear()

    def terminal_logs(self, step):
        print("[INFO]\tSteps: {}\tMean Reward: {:0.3f}\tStd of Reward: {:0.3f}".format(step, np.mean(self.rewards), np.std(self.rewards)))

    def tensorboard_logs(self, step):
        self.writer.add_scalar('Environment/Cumulative_reward', np.mean(self.rewards), step)
        self.writer.add_scalar('Environment/Episode_length', np.mean(self.episode_lens), step)

        self.writer.add_scalar('Learning_rate/Actor_model', np.mean(self.actor_lrs), step)
        self.writer.add_scalar('Learning_rate/Critic _model', np.mean(self.critic_lrs), step)

        self.writer.add_scalar('Loss/Policy_loss', np.mean(self.actor_losses), step)
        self.writer.add_scalar('Loss/Value_loss', np.mean(self.critic_losses), step)

        self.writer.add_scalar('Policy/Value_estimate', np.mean(self.advantages), step)
    
        self.clear_memory()
