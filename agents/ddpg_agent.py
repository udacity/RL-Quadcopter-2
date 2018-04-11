from pathlib import Path
import numpy as np
import tensorflow as tf

from .base_agent import BaseAgent

from .replay_buffer import ReplayBuffer
from .ddpg_actor import Actor
from .ddpg_critic import Critic
from .noise import OUNoise, OUNoise2


class DeepDPGAgent(BaseAgent):
    batch_size = 64
    tau = 0.001
    gamma = 0.99
    learning_rate = 0.0001

    """Implement Deep DPG control agent

    From paper by Lillicrap, Timothy P. "Continuous Control with Deep Reinforcement Learning."
    https://arxiv.org/pdf/1509.02971.pdf

    """
    def __init__(self, task, replay_buffer_size=100000, batch_size=None):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        super().__init__(task)

        # Create actor and critic
        self.critic = Critic(task, learning_rate=DeepDPGAgent.learning_rate * 100)
        self.actor = Actor(task, self.critic, learning_rate = DeepDPGAgent.learning_rate)

        #self.noise = OUNoise2(
        #    task.num_actions,
        #    theta=0.15,
        #    sigma=0.2)
        self.noise = OUNoise(
            task.num_actions,
            theta=0.15,
            sigma=25)

        # Create critic NN

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.actor.set_session(self.session)
        self.critic.set_session(self.session)

        self.actor.initialize()
        self.critic.initialize()

        # writer = tf.summary.FileWriter('graph', self.session.graph)

        self.prev_state = None

        self.batch_size = batch_size or DeepDPGAgent.batch_size
        self.tau = DeepDPGAgent.tau
        self.gamma = DeepDPGAgent.gamma

        self.best_score = None
        self.episode_score = 0.0
        self.episode_ticks = 0

        self.episode = 1

        self.saver = tf.train.Saver()

        self.load_task_agent()

    def reset_episode(self):
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.noise.reset()
        return self.task.reset()

    def act(self, state):
        action = self.actor.get_action(np.expand_dims(state, axis=0))[0]

        noise = self.noise.sample()
        for i in range(self.task.num_actions):
            noise[i] = min(self.task.action_high, max(self.task.action_low, noise[i]))

        action += noise

        return action

    def act_target(self, state):
        action = self.actor.get_target_action(np.expand_dims(state, axis=0))[0]
        return action

    def step(self, action, reward, next_state, done):
        self.episode_ticks += 1
        self.episode_score += reward

        if self.prev_state is not None:
            self.replay_buffer.add([self.prev_state, action, reward, next_state, done])

        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)

            prev_states = np.array([t[0] for t in batch])
            prev_actions = np.array([t[1] for t in batch])
            rewards = np.expand_dims(np.array([t[2] for t in batch]), axis=1)
            states = np.array([t[3] for t in batch])

            y = rewards + self.gamma * self.critic.get_target_value(states, self.actor.get_target_action(states))
            self.critic.learn(prev_states, prev_actions, y)
            self.actor.learn(prev_states)

            self.critic.update_target(self.tau)
            self.actor.update_target(self.tau)

        if done:
            if self.best_score is not None:
                self.best_score = max(self.best_score, self.episode_score)
            else:
                self.best_score = self.episode_score
            self.episode += 1
            self.save_task_agent()

        self.prev_state = next_state if not done else None

    def show_episode_stats(self):
        print("Deep DPG episode stats: t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
            self.episode_ticks, self.episode_score, self.best_score, 0))# self.noise_scale))  # [debug]

    @property
    def noise_scale(self):
        return 0

    def load(self, path):
        self.saver.restore(self.session, path)

    def save(self, path):
        self.saver.save(self.session, path)


class DeepDPGPlayer(BaseAgent):
    def __init__(self, task):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        super().__init__(task)

        # Create actor and critic
        self.critic = Critic(task, learning_rate = DeepDPGAgent.learning_rate * 10)
        self.actor = Actor(task, self.critic, learning_rate = DeepDPGAgent.learning_rate)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.actor.set_session(self.session)
        self.critic.set_session(self.session)

        self.actor.initialize()
        self.critic.initialize()

        self.best_score = None
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.reset_episode_vars()
        self.prev_state = None

        self.episode = 1

        self.saver = tf.train.Saver()

        self.num_actions = task.num_actions
        self.num_states = task.num_states

        self.load_task_agent()

    def reset_episode_vars(self):
        self.episode_score = 0.0
        self.episode_ticks = 0

    def act(self, state):
        return self.actor.get_target_action(np.expand_dims(state, axis=0))[0]

    def step(self, action, reward, next_state, done):
        """Process state, reward, done flag, and return an action.

        Params
        ======
        - state: current state vector as NumPy array, compatible with task's state space
        - reward: last reward received
        - done: whether this episode is complete

        Returns
        =======
        - action: desired action vector as NumPy array, compatible with task's action space
        """
        self.episode_ticks += 1
        self.episode_score += reward

        action_str = ', '.join('{:7.3f}'.format(a) for a in action)
        position_str = ', '.join('{:7.3f}'.format(a) for a in self.prev_state[0:6])

        print('[{}] => [{}], R = {:7.3f}'.format(position_str, action_str, reward))

        self.prev_state = next_state

        # Output some information at the end of the episode
        if done:
            if self.best_score is not None:
                self.best_score = max(self.best_score, self.episode_score)
            else:
                self.best_score = self.episode_score
            print("Deep DPG episode stats: t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                self.episode_ticks, self.episode_score, self.best_score, 0))# self.noise_scale))  # [debug]
            self.reset_episode_vars()
            self.episode += 1

        return action

    def reset_episode(self):
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.prev_state = self.task.reset()
        return self.prev_state

    @property
    def noise_scale(self):
        return 0

    def load(self, path):
        self.saver.restore(self.session, path)

    def save(self, path):
        # Can't save anything
        pass
