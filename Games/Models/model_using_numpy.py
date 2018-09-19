import gym
import numpy as np
import _pickle as pickle


class model_using_numpy():

    def __init__(self, num_hidden_layer_neurons, input_dimensions, modelFileName, isResume):
        
        # fixed hyperparameters
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99
        self.learning_rate = 1e-4

        self.batch_size = 10
        self.saveFreq = 20

        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.input_dimensions = input_dimensions

        self.expectation_g_squared = {}
        self.g_dict = {}

        self.modelFileName = modelFileName
        
        if isResume is True and modelFileName is not None:
            self.weights = pickle.load(open(modelFileName, 'rb'))
        else:
            self.weights = {
                '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
                '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
            }

        for layer_name in self.weights.keys():
            self.expectation_g_squared[layer_name] = np.zeros_like(self.weights[layer_name])
            self.g_dict[layer_name] = np.zeros_like(self.weights[layer_name])
            
        

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def relu(self, vector):
        vector[vector < 0] = 0
        return vector

    def apply_neural_nets(self, observation_matrix):
        """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
        hidden_layer_values = np.dot(self.weights['1'], observation_matrix)
        hidden_layer_values = self.relu(hidden_layer_values)
        output_layer_values = np.dot(hidden_layer_values, self.weights['2'])
        output_layer_values = self.sigmoid(output_layer_values)
        return hidden_layer_values, output_layer_values

    def compute_gradient(self, gradient_log_p, hidden_layer_values, observation_values, weights):
        delta_L = gradient_log_p
        dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
        delta_l2 = np.outer(delta_L, weights['2'])
        delta_l2 = self.relu(delta_l2)
        dC_dw1 = np.dot(delta_l2.T, observation_values)
        return {
            '1': dC_dw1,
            '2': dC_dw2
        }

    def update_weights(self, weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
        epsilon = 1e-5
        for layer_name in weights.keys():
            g = g_dict[layer_name]
            expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
            weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
            g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

    def discount_rewards(self, rewards, gamma):
        """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
        This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def discount_with_rewards(self, gradient_log_p, episode_rewards, gamma):
        """ discount the gradient with the normalized rewards """
        discounted_episode_rewards = self.discount_rewards(episode_rewards, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return gradient_log_p * discounted_episode_rewards

    def trainModel(self, episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards, episode_number):
         # Combine the following values for the episode
        episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
        episode_observations = np.vstack(episode_observations)
        episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
        episode_rewards = np.vstack(episode_rewards)

        # Tweak the gradient of the log_ps based on the discounted rewards
        episode_gradient_log_ps_discounted =self.discount_with_rewards(episode_gradient_log_ps, episode_rewards, self.gamma)
        gradient = self.compute_gradient(
            episode_gradient_log_ps_discounted,
            episode_hidden_layer_values,
            episode_observations,
            self.weights)

            # Sum the gradient for use when we hit the batch size
        for layer_name in gradient:
            self.g_dict[layer_name] += gradient[layer_name]
        
        if episode_number % self.batch_size == 0:
            self.update_weights(self.weights, self.expectation_g_squared, self.g_dict, self.decay_rate, self.learning_rate)

        if episode_number % self.saveFreq == 0:
            self.saveWeights()
            
    def saveWeights(self):
        pickle.dump(self.weights, open(self.modelFileName, 'wb'))