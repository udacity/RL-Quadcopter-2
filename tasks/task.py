import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(self.state)

        self.action_low = 0
        self.action_high = 900

        self.simplified = True

        if self.simplified:
            self.action_size = 1
        else:
            self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.task_name = 'default'

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    @property
    def num_states(self):
        return self.state_size

    @property
    def num_actions(self):
        return self.action_size

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []

        if self.simplified:
            rotor_speeds = np.array([rotor_speeds[0]] * 4)

        for _ in range(self.action_repeat):
            self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            state_all.append(self.state)
        next_state = np.concatenate(state_all)
        return next_state, reward, self.done

    @property
    def state(self):
        return np.append(self.sim.pose, self.sim.v)

    @property
    def position(self):
        return self.sim.pose[0:3]

    @property
    def orientation(self):
        return self.sim.pose[3:6]

    @property
    def timeout(self):
        return self.sim.done

    @property
    def done(self):
        return self.timeout or self.is_task_finished()

    def is_task_finished(self):
        return False

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.reset_vars()
        state = np.concatenate([self.state] * self.action_repeat)
        return state

    def reset_vars(self):
        pass