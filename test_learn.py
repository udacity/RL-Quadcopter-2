import sys
import numpy as np
import tensorflow as tf

from tasks import Takeoff, Hover
from agents import DeepDPGAgent

# Setup learning parameters
DeepDPGAgent.tau = 0.001
DeepDPGAgent.gamma = 0.99
DeepDPGAgent.learning_rate = 0.0001

# Create task and agent
# task = Takeoff()
task = Hover()
task.task_name = 'emulate_' + task.task_name

agent = DeepDPGAgent(task, batch_size=128)

num_episodes = 2#000


def learn_episode(i_episode, agent, task):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        if done:
            print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.episode_score, agent.best_score, agent.noise_scale))  # [debug]
            break


def play_episode(i_episode, agent, task):
    state = agent.reset_episode()
    eposide_score = 0.0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)

        action_str = ', '.join('{:7.3f}'.format(a) for a in action)
        position_str = ', '.join('{:7.3f}'.format(a) for a in state[0:6])
        print('{:5.2f}: [{}] => [{}], R = {:7.3f}'.format(task.sim.time, position_str, action_str, reward))

        eposide_score += reward
        state = next_state
        if done:
            print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, eposide_score, agent.best_score or 0, agent.noise_scale))  # [debug]
            break


for i_episode in range(num_episodes):
    if i_episode % 40 == 0:
        play_episode(i_episode, agent, task)

    learn_episode(i_episode, agent, task)

    sys.stdout.flush()

