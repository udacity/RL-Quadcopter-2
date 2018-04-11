import math
import numpy as np
from .task import Task

class Hover(Task):
    """Hover task."""
    def __init__(self, target_z=150., start_z=100., hover_duration=3.0):
        super().__init__(
            target_pos=np.array([0., 0., target_z]),
            init_pose=np.array([0., 0., start_z, 0., 0., 0.]),
            runtime=10.0
        )

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = target_z
        self.hover_duration = hover_duration

        # Reward task-specific variables
        self.last_time = 0.0
        self.time_hover = 0.0

        self.task_name = 'hover'

    def is_task_finished(self):
        return self.time_hover >= self.hover_duration

    def get_reward(self):
        reward = 0.0

        # Compute reward for reaching target height
        target_distance = abs(self.target_z - self.position[2])
        height_reward = min(target_distance, 50.0)
        reward -= 1.0 * height_reward

        # Penalty for getting too low.
        # Simulator do not have "ground", so task is over when copter gets below zero. Thus getting too low should be punished
        # too_low_threshold = 3.0
        # too_low_penalty = math.exp(too_low_threshold - math.pow(self.position[2], 4.0)) if self.position[2] < too_low_threshold else 0
        # reward -= too_low_penalty

        # Penalty of going too far from the target
        too_far_threshold = 25.
        too_far_penalty = target_distance - too_far_threshold if target_distance > too_far_threshold else 0.0
        too_far_penalty = math.pow(too_far_penalty, 1.1)
        reward -= too_far_penalty * 1.0

        # Penalize for shifting in horizontal plane
        # dist_penalty = math.sqrt(self.position[0] * self.position[0] + self.position[1] * self.position[1])
        # reward -= dist_penalty * 2.0

        # Penalise for rotating
        # rotate_penalty = np.abs(self.sim.angular_v).sum()
        # reward -= rotate_penalty * 1.0

        # Orientation reward
        orientation_penalty = (1 - np.cos(self.orientation)).sum()
        reward -= orientation_penalty * 1.0

        if abs(self.position[2] - self.target_z) < 0.5:
            self.time_hover += self.sim.time - self.last_time
            reward += 5.0 * self.time_hover

        if self.is_task_finished():  # agent stayed around hover position for long enough
            reward += 50.0  # bonus reward
            done = True
        elif self.timeout:  # agent has run out of time
            reward -= 50.0  # extra penalty
            done = True

        self.last_time = self.sim.time

        return reward

    def reset_vars(self):
        self.last_time = self.sim.time
        self.time_hover = 0.0
