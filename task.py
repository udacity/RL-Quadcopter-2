import numpy as np
from physics_sim import PhysicsSim
import random
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
        self.action_repeat = 2

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
        
        reward_x = -min(abs(self.target_pos[0] - self.sim.pose[0]), 20.0)
        reward_y = -min(abs(self.target_pos[1] - self.sim.pose[1]), 20.0)
        reward_z = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)
     
        angular_stationary_score = (10.-.3*np.linalg.norm(self.sim.angular_v)**2)
     
       
        reward = reward_x + reward_y + (reward_z * 1.5) + (angular_stationary_score / 10) + self.sim.v[2]
        #reward = reward_z + (angular_stationary_score / 10) + self.sim.v[2]
        
        
        if( self.sim.v[2]> .1):
            reward += 1
            
        if( self.sim.v[2]> .3):
            reward += 5
            
        if( self.sim.v[2]> .5):
            reward += 5
            
        if (self.sim.v[2]< .0):
            reward -= 5
            
        if (self.sim.v[2]< -.3):
            reward -= 10
        
        if(self.sim.pose[2] >= self.target_pos[2]):
            reward += 100
    
        init_distance = 80
        
        if( abs(self.sim.pose[2] - self.target_pos[2]) < init_distance):
            reward += 10
        else:
            reward -= 10
            
        if( abs(self.sim.pose[2] - self.target_pos[2]) < init_distance * .8):
            reward += 50
        
        if( abs(self.sim.pose[2] - self.target_pos[2]) < init_distance * .7):
            reward += 50
            
        if( abs(self.sim.pose[2] - self.target_pos[2]) < init_distance * .6):
            reward += 50
    
        if(random.randint(0,100)<5):
            print("positions (x,y,z), reward:",self.sim.pose[:3],reward)
            
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