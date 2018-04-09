import numpy as np
import tensorflow as tf

from .utils import scope_variables_mapping


class Critic:
    def __init__(self, task, scope_name='critic', learning_rate=0.001):
        self.scope = scope_name

        self.input_states = tf.placeholder(
            tf.float32,
            (None, task.num_states),
            name='critic/states')

        self.input_actions = tf.placeholder(
            tf.float32,
            (None, task.num_actions),
            name='critic/actions')

        self.is_training = tf.placeholder(tf.bool, name='critic/is_training')

        self.target = self.create_model(self.input_states, self.input_actions, task, scope_name + '_target')
        self.current = self.create_model(self.input_states, self.input_actions, task, scope_name + '_current', training=self.is_training)

        self.y = tf.placeholder(tf.float32, (None, 1), name='critic/y')
        loss = tf.losses.mean_squared_error(self.y, self.current)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        self.tau = tf.placeholder(tf.float32, name='critic/tau')
        self.assignments = [tf.assign(t, c * self.tau + (1-self.tau) * t)
                            for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.init = [tf.assign(t, c)
                     for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.session = None

    def initialize(self):
        self.session.run(self.init)

    def create_model(self, input_states, input_actions, task, scope_name, training=False, reuse=False):

        with tf.variable_scope(scope_name, reuse=reuse):
            g = 0.0001
            # 2 layers of states
            dense_s = tf.layers.dense(input_states, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense_s = tf.layers.dropout(dense_s, 0.5, training=training)
            # dense_s = tf.nn.l2_normalize(dense_s, epsilon=0.01)
            # dense_s = tf.layers.batch_normalization(dense_s, training=training)

            dense_s = tf.layers.dense(dense_s, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense_s = tf.layers.dropout(dense_s, 0.5, training=training)
            # dense_s = tf.nn.l2_normalize(dense_s, epsilon=0.01)
            # dense_s = tf.layers.batch_normalization(dense_s, training=training)

            # One layer of actions
            dense_a = tf.layers.dense(input_actions, 32,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense_a = tf.layers.dropout(dense_a, 0.5, training=training)
            # dense_a = tf.nn.l2_normalize(dense_a, epsilon=0.01)
            # dense_a = tf.layers.batch_normalization(dense_a, training=training)

            # Merge together
            dense = tf.concat([dense_s, dense_a], axis=1)

            # Decision layer
            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=0.01)
            # dense = tf.layers.batch_normalization(dense, training=training)

            # Output layer
            dense = tf.layers.dense(dense, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-g, maxval=g),
                                    bias_initializer=tf.random_uniform_initializer(minval=-g, maxval=g))
            result = dense

        return result

    def set_session(self, session):
        self.session = session

    def get_value(self, state, action):
        return self.session.run(
            self.current,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def get_target_value(self, state, action):
        return self.session.run(
            self.target,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def learn(self, states, actions, targets):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input_states: states,
                self.input_actions: actions,
                self.y: targets,
                self.is_training: True})

    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})
