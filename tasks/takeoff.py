"""Takeoff task."""

import math
from pathlib import Path
import numpy as np
from .task import Task

class Takeoff(Task):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self, target_z=20., start_z=5.):
        super().__init__(
            target_pos=np.array([0., 0., target_z]),
            init_pose=np.array([0., 0., start_z, 0., 0., 0.])
        )
        print("Starting takeoff task")

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = target_z

        self.task_name = 'takeoff'

    def is_task_finished(self):
        return self.position[2] >= self.target_z

    def get_reward(self):
        reward = 0.0

        # Compute reward for reaching target height
        height_reward = min(abs(self.target_z - self.position[2]), 50.0)
        reward -= height_reward * height_reward

        # Penalize for shifting in horizontal plane
        # dist_penalty = math.sqrt(self.position[0] * self.position[0] + self.position[1] * self.position[1])
        # reward -= dist_penalty * 2.0

        # Penalise for rotating
        # rotate_penalty = np.abs(self.sim.angular_v).sum()
        # reward -= rotate_penalty * 1.0

        # Orientation reward
        orientation_penalty = (1 - np.cos(self.orientation)).sum()
        reward -= orientation_penalty * 1.0

        if self.is_task_finished():  # agent has crossed the target height
            reward += 50.0 # bonus reward for reaching the target
        elif self.timeout:  # agent has run out of time
            reward -= 500.0  # extra penalty for timeout or going out of emulated space

        return reward
