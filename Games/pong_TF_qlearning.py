import gym
import numpy as np
import time
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

import calendar

import tensorflow as tf
from models.model_using_tensorflow import Tensorflow

import sys,getopt

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

def saveFile(reward_sum, timestamps):
    np.savetxt("history/pong_tf_qlearning/pong_tf_qlearning_rewards.txt",reward_sum, fmt= '%d')

    currentTime = calendar.timegm(time.gmtime())
    timestamps.append(currentTime)
    np.savetxt("history/pong_tf_qlearning/pong_tf_qlearning_timestamps.txt",timestamps, fmt= '%d')

def loadFile():
    Rewards = np.loadtxt("history/pong_tf_qlearning/pong_tf_qlearning_rewards.txt", dtype=int)
    timestamps = np.loadtxt("history/pong_tf_qlearning/pong_tf_qlearning_timestamps.txt", dtype=int)
    return Rewards.tolist(), timestamps.tolist()

def visualize(number_eps, rewards):
    plt.plot(number_eps, rewards, linestyle='--')
    plt.ylim(-25,20)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.show()

    yhat = savgol_filter(rewards, 41, 2)

    plt.plot(number_eps,yhat)
    plt.show()


def startTraining(isResume, isRender):
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    #hyper-parameters
    input_dimensions = 80 * 80
    num_hidden_layer_neurons = 200
    
    number_of_episodes = 100000
    saveFreq = 500
    modelChkpntFreq = 10000

    #Initialising attriobutes
    prev_processed_observations = None
    running_reward = None
    reward_sum = 0
    
    resume = isResume
    render = isRender

    if resume is True:
        reward_sum_array, timestamps = loadFile()
    else:
        reward_sum_array, timestamps = [], []

    episode_number = len(reward_sum_array)

    model = Tensorflow(num_hidden_layer_neurons, input_dimensions, "pong_TF_qlearning_weights.ckpt", 'history/pong_tf_qlearning/', resume)
        
    episode_observations, episode_rewards, episode_actions = [], [], []

    while episode_number < number_of_episodes:
        if render:
            env.render()        
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        # hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        up_probability = model.predict(processed_observations)
        
        # episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)
    
        reward_sum += reward

        if action == 2:
            action = 1
        else:
            action = 0

        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_observations.append(processed_observations)

        if done: 
            episode_number += 1
            
            model.trainNetwork(episode_rewards, episode_observations, episode_actions)
            
            episode_observations, episode_rewards, episode_actions = [], [], [] # reset values
            
            observation = env.reset() # reset env

            reward_sum_array.append(reward_sum)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print ('Episode Number %d resetting. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))

            reward_sum = 0
            prev_processed_observations = None
            
            if episode_number % saveFreq == 0:
                model.saveModel()
                saveFile(reward_sum_array, timestamps)

            if episode_number % modelChkpntFreq == 0:
                filename = 'pong_TF_qlearning_weights_' + str(episode_number) + '.ckpt'
                model.saveCheckpoint(filename)

    env.close()

def demoFromCheckpoint(episode_number):
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image
    prev_processed_observations = None

    #hyper-parameters
    input_dimensions = 80 * 80
    num_hidden_layer_neurons = 200

    resume = True

    #Create a model object
    if episode_number == 0:
        filename = 'pong_TF_qlearning_weights.ckpt'
    else:
        filename = 'pong_TF_qlearning_weights_episode_' + str(episode_number) + '.ckpt'

    model = Tensorflow(num_hidden_layer_neurons, input_dimensions, filename, 'history/pong_tf_qlearning/', resume)

    reward_sum = 0
    episode_number = 0

    while episode_number < 10:
        time.sleep(0.01)
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        up_probability = model.predict(processed_observations)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)
    
        reward_sum += reward

        # carry out the chosen action
        observation, reward, done, _ = env.step(action)
        reward_sum += reward

        if done :
            episode_number += 1

            observation = env.reset() # reset env
            print ('Episode Number %d. episode reward total was %f.' % (episode_number, reward_sum))

            reward_sum = 0
            prev_processed_observations = None
            
    env.close()


def plotRewards():
    reward_sum_array, timestamps = loadFile()
    number_eps = np.arange(len(reward_sum_array))
    visualize(number_eps, reward_sum_array)

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"ht:r:R:g:d:",["help","train=","resume=","render=","graph=","demochkpt="])
except getopt.GetoptError:
    print('python3 pong_TF_qlearning.py --train True --resume True --render False --graph True')
    sys.exit(2)

isGraph = False
demoChkpt = 0

for opt, arg in opts:
    if opt == '-h':
        print('python3 pong_numpy_qlearning.py --train True --resume True --render False')
        sys.exit()
    elif opt in ("-t", "--train"):
        isTrain = arg
    elif opt in ("-r", "--resume"):
        isResume = arg
    elif opt in ("-R", "--render"):
        isRender = arg
    elif opt in ("-g", "--graph"):
        isGraph = arg
    elif opt in ("-d", "--demochkpt"):
        demoChkpt = arg

if isGraph == 'True':
    plotRewards()

if isTrain == 'True':
    startTraining(isResume == 'True', isRender == 'True')
else:
    print('Starting from demo checkpoint : ', demoChkpt)
    demoFromCheckpoint(demoChkpt)