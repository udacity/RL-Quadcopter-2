import numpy as np
from physics_sim import PhysicsSim
import sys

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

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # simple reward
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # sigmoid reward
        #sig_reward = 1 - self.sigmoid(sum(abs(self.sim.pose[:3] - np.float32(self.target_pos))) * .3)
        #print(sig_reward)
        
        # tanh reward
        #tanh_reward = 1 - (np.tanh((abs(self.sim.pose[:3] - self.target_pos))).sum() * 0.3)
        
        reward_x = np.tanh(1 - 0.03*(abs(self.sim.pose[0] - self.target_pos[0])))
        reward_y = np.tanh(1 - 0.03*(abs(self.sim.pose[1] - self.target_pos[1])))
        reward_z = np.tanh(1 - 0.03*(abs(self.sim.pose[2] - self.target_pos[2])))
       
        reward = reward_x + reward_y + reward_z
    
    
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state