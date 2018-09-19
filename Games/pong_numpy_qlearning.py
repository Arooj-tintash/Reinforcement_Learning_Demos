import gym
import numpy as np
from matplotlib import pyplot as plt


from models.model_using_numpy import model_using_numpy

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
    np.savetxt("history/pong_numpy_qlearning_rewards.txt",reward_sum, fmt= '%d')

def loadFile():
    Rewards = np.loadtxt("history/pong_numpy_qlearning_rewards.txt", dtype=int)
    return Rewards.tolist()

def visualize(number_eps, rewards):
    plt.plot(number_eps, rewards, linestyle='--')
    plt.ylim(-25,20)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.show()


def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    #hyper-parameters
    input_dimensions = 80 * 80
    num_hidden_layer_neurons = 200
    number_of_episodes = 100

    #Initialising attriobutes
    prev_processed_observations = None
    running_reward = None
    reward_sum = 0
    
    resume = True
    render = False

    if resume is True:
        reward_sum_array = loadFile()
    else:
        reward_sum_array = np.array

    episode_number = len(reward_sum_array)

    #Create a model object
    model = model_using_numpy(num_hidden_layer_neurons, input_dimensions, "history/pong_numpy_qlearning_weights.p", resume)

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while episode_number < number_of_episodes:
        if render:
            env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = model.apply_neural_nets(processed_observations)
    
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, _ = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)
        
        if done:
            episode_number += 1
            
            reward_sum_array.append(reward_sum)

            model.trainModel(episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards, episode_number)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print ('Episode Number %d resetting. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
            
            reward_sum = 0
            prev_processed_observations = None

            if episode_number % 10 == 0:
                saveFile(reward_sum_array)
        
main()