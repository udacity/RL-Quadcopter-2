import numpy as np

from pathlib import Path

"""Generic base class for reinforcement learning agents."""

class BaseAgent:
    """Generic base class for reinforcement reinforcement agents."""

    def __init__(self, task):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        self.task = task

    def act(self, state):
        return np.array([0] * self.task.num_actions)


    def step(self, state, reward, done):
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
        raise NotImplementedError("{} must override step()".format(self.__class__.__name__))

    def load(self, path):
        pass

    def save(self, path):
        pass

    def load_task_agent(self):
        # Load agent for task from file if it exists
        path = Path.cwd() / 'models'
        file_name = self.task.task_name + '.model'
        if (path / (file_name + '.index')).exists():
            self.load(str(path / file_name))

    # Save agent model for the task
    def save_task_agent(self):
        path = Path.cwd() / 'models'
        path.mkdir(exist_ok=True)
        file_name = self.task.task_name + '.model'
        self.save(str(path / file_name))

