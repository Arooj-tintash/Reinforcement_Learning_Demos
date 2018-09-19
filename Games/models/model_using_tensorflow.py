# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import tensorflow as tf
import os


class Tensorflow():
    def __init__(self, num_hidden_layer_neurons, input_dimensions, modelFileName, modelDir, isResume):
        #Fixed Hyperparameters
        self.learning_rate = 0.0005
        self.gamma = 0.99 # discount factor for reward

        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.input_dimensions = input_dimensions
        self.batch_size = 1
        self.saveFreq = 20

        self.session = tf.InteractiveSession()

        self.observation_placeholder = tf.placeholder(tf.float32,
                                            [None, input_dimensions])

        hidden_layer = tf.layers.dense(
            self.observation_placeholder,
            units=num_hidden_layer_neurons,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(
            hidden_layer,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.output_layer,
            weights=self.advantage)
            
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)

        tf.global_variables_initializer().run(session = self.session)

        self.modeldir = modelDir
        self.checkpoint_file = os.path.join(modelDir,modelFileName)
        self.saver = tf.train.Saver()

        if isResume:
            self.saver.restore(self.session, self.checkpoint_file)


    def discount_with_rewards(self, episode_rewards):
        discounted_rewards = np.zeros_like(episode_rewards)
        for t in range(len(episode_rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(episode_rewards)):
                discounted_reward_sum += episode_rewards[k] * discount
                discount *= self.gamma
                if episode_rewards[k] != 0:
                    # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[t] = discounted_reward_sum

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    def predict(self, processed_observation):
        up_probability = self.session.run(
            self.output_layer,
            feed_dict={self.observation_placeholder: processed_observation.reshape([1, -1])}) #Review later-------------------------------------------------------------

        return up_probability
    
    def trainNetwork(self, episode_rewards, episode_observations, episode_actions):
        # if episode_number % batch_size == 0:
        discounted_reward = self.discount_with_rewards(episode_rewards)
        # update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

        states_stack = np.vstack(episode_observations)
        actions_stack = np.vstack(episode_actions)
        rewards_stack = np.vstack(discounted_reward)

        feed_dict = {
            self.observation_placeholder: states_stack,
            self.sampled_actions: actions_stack,
            self.advantage: rewards_stack
        }
        self.session.run(self.train_op, feed_dict)

    def saveModel(self):
        self.saver.save(self.session, self.checkpoint_file)

    def saveCheckpoint(self, fileName):
        checkpoint_file = os.path.join(self.modeldir, fileName)
        self.saver.save(self.session, checkpoint_file)