"""Takeoff task."""

import math
from pathlib import Path
import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from .quadrocopter_task import QuadrocopterTask

class Takeoff(QuadrocopterTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self, num_actions=1):
        super().__init__(num_actions)
        print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]
        print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.task_name = 'takeoff'

    def reset(self):
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def calculate_reward(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = -min(abs(self.target_z - pose.position.z), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        dist = math.sqrt(pose.position.x * pose.position.x + pose.position.y * pose.position.y)
        reward -= dist * 2.0
        if pose.position.z >= self.target_z:  # agent has crossed the target height
            reward += 50.0 - dist * 5.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 50.0  # extra penalty
            done = True

        return reward, done
