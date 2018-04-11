"""Takeoff task."""

import math
from pathlib import Path
import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from .task import Task

class QuadrocopterTask(Task):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self, num_actions=1):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        assert num_actions == 1 or num_actions == 3
        self.num_actions = num_actions
        self.num_states = 6

    def set_agent(self, agent):
        """Set an agent to carry out this task; to be called from update."""
        super().set_agent(agent)
        self.load_agent()

    def load_agent(self):
        # Load agent from file if it exists
        path = Path.cwd() / 'models'
        file_name = self.task_name + '.model'
        if (path / (file_name + '.index')).exists():
            self.agent.load(str(path / file_name))

    def save_agent(self):
        path = Path.cwd() / 'models'
        path.mkdir(exist_ok=True)
        file_name = self.task_name + '.model'
        self.agent.save(str(path / file_name))

    @property
    def state_size(self):
        return self.num_states

    @property
    def action_size(self):
        return self.num_actions

    def calculate_reward(self, timestamp, pose, angular_velocity, linear_acceleration):
        return 0.0, False

    def make_state(self, pose, angular_velocity, linear_acceleration):
        return np.array([pose.position.x, pose.position.y, pose.position.z,
                         linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])


    def filter_action(self, action):
        return np.clip(action.flatten(), self.action_space.low[0:self.num_actions],
                       self.action_space.high[0:self.num_actions])  # flatten, clamp to action space limits

    def extract_action(self, action):
        a = self.filter_action(action)
        # return np.array([0.0, 0.0, a[0]])
        return a

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = self.make_state(pose, angular_velocity, linear_acceleration)

        # Compute reward / penalty and check if this episode is complete
        reward, done = self.calculate_reward(timestamp, pose, angular_velocity, linear_acceleration)

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        if done:
            self.save_agent()
        else:
            # print(' => [{:7.3f}, {:7.3f}, {:7.3f}]'.format(action[0], action[1], action[2]), end='')
            # print(' R = {:7.3f}'.format(reward), end='\n')
            pass

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = self.extract_action(action)

            # Make wrench force vector depending on number of possible actions
            if self.num_actions == 3:
                force_vector = Vector3(action[0], action[1], action[2])
            elif self.num_actions == 1:
                force_vector = Vector3(0.0, 0.0, action[0])
            # Torque is just a zero vector, we don't control it
            torque_vector = Vector3(0, 0, 0)

            return Wrench(
                    force=force_vector,
                    torque = torque_vector
                ), done
        else:
            return Wrench(), done
