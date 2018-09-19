import gym
import numpy as np
from matplotlib import pyplot as plt


from Models.model_using_numpy import *

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3

def saveFile(reward_sum):
    np.savetxt("History/pong_numpy_qlearning_rewards.txt",reward_sum, fmt= '%d')

def loadFile():
    Rewards = np.loadtxt("History/pong_numpy_qlearning_rewards.txt", dtype=int)
    return Rewards

def visualize(number_eps, rewards):
    plt.plot(number_eps, rewards, linestyle='--')
    plt.ylim(-25,20)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.show()


def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    reward_sum = 0
    reward_sum_array = []
    expectation_g_squared = {}
    g_dict = {}
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    weights = create_neural_network(num_hidden_layer_neurons, input_dimensions)
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, model_using_numpy.prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
    
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)
        reward_sum_array.append(reward_sum)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)
        
        if done: 
            model_using_numpy.episode_number += 1
            
             # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, model_using_numpy.gamma)
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights)

             # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]
            
            if model_using_numpy.episode_number % model_using_numpy.batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, model_using_numpy.decay_rate, model_using_numpy.learning_rate)
            
            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            running_reward = reward_sum if model_using_numpy.running_reward is None else model_using_numpy.running_reward * 0.99 + reward_sum * 0.01
            print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            prev_processed_observations = None

    saveFile(reward_sum_array)



        
main()