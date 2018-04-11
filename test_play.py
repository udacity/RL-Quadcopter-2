import sys
import numpy as np
import tensorflow as tf

from tasks import Takeoff, Hover
from agents import DeepDPGPlayer

# Create task and agent
task = Takeoff()
#task = Hover()
task.task_name = 'emulate_' + task.task_name

agent = DeepDPGPlayer(task)

num_episodes = 1

for i_episode in range(num_episodes):
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
    sys.stdout.flush()

