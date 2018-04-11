import numpy as np
import tensorflow as tf

from .utils import scope_variables_mapping


class Actor:
    def __init__(self, task, critic, scope_name='actor', learning_rate=0.001):
        self.input = tf.placeholder(tf.float32, (None, task.num_states), name='actor/states')
        self.is_training = tf.placeholder(tf.bool, name='actor/is_training')

        self.target = self.create_model(self.input, task, scope_name + '_target')
        self.current = self.create_model(self.input, task, scope_name + '_current', self.is_training)

        self.q_gradients = tf.placeholder(tf.float32, (None, task.num_actions))

        critic = critic.create_model(self.input, self.current, task, critic.scope + '_current', reuse=True)
        loss = tf.reduce_mean(-critic)
        tf.losses.add_loss(loss)

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = optimizer.minimize(loss, var_list=tf.trainable_variables(scope_name + '_current'))

        self.tau = tf.placeholder(tf.float32)
        self.assignments = [tf.assign(t, c * self.tau + (1-self.tau) * t)
                            for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.init = [tf.assign(t, c)
                     for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.session = None

    def initialize(self):
        self.session.run(self.init)

    def create_model(self, inputs, task, scope_name, training=False):
        g = 0.001
        eps = 1
        with tf.variable_scope(scope_name):
            dense = tf.layers.dense(inputs, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, task.num_actions,
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-g, maxval=g),
                                    bias_initializer=tf.random_uniform_initializer(minval=-g, maxval=g))

            action_min = np.array([task.action_low] * task.num_actions)
            action_max = np.array([task.action_high] * task.num_actions)
            action_range = action_max - action_min

            result = dense * action_range + action_min

        return result

    def set_session(self, session):
        self.session = session

    def get_action(self, state):
        return self.session.run(
            self.current,
            feed_dict={
                self.input: state,
                self.is_training: False})

    def get_target_action(self, state):
        return self.session.run(self.target, feed_dict={self.input: state})

    def learn(self, state):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input: state,
                self.is_training: True})

    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})
