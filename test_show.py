import tensorflow as tf
import numpy as np

from tasks import Takeoff, Hover
from agents import DeepDPGPlayer

task = Hover()
task.task_name = 'emulate_' + task.task_name

agent = DeepDPGPlayer(task)

print(task.num_states)

a_values = [0.0, 200.0, 500.0, 800.]

print('z-pos: ', end='')
for a in a_values:
    print(' a = {:7.0f}  |'.format(a), end='')
print('')

for s in range(100, 200, 10):
    print('@{:4.1f}: '.format(s), end='')
    for a in a_values:
        states = np.array([[0, 0, s, 0, 0, 0, 0, 0, 0] * 3])
        actions = np.array([[a]])

        q = agent.critic.get_target_value(states, actions)[0]

        print('   {:9.2f}  |'.format(q[0]), end='')

    print('')