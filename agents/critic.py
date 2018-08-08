from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.0002
        
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        net_states = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)

        net_states = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)
        
        net_actions = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)

        net_actions = layers.Dense(units=256, activation=None, use_bias=False, kernel_regularizer=keras.initializers.glorot_normal(seed=None))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', kernel_regularizer=keras.initializers.glorot_normal(seed=None))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        