import gym
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from models.model_using_DQN import model_using_DQN

from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

import time
import calendar

import sys, getopt

# get action from model using epsilon-greedy policy
def get_action(history, agent):
    history = np.float32(history / 255.0)
    if np.random.rand() <= agent.epsilon:
        return random.randrange(agent.action_size)
    else:
        q_value = agent.model.predict(history)
        return np.argmax(q_value[0])

# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    gray = rgb2gray(observe)
    # cropped_frame = gray[8:-12,4:-12]
    normalized_frame = gray/255.0
    processed_observe = np.uint8(resize(normalized_frame, [110,84]))
    # processed_observe = np.uint8(
    #     resize(rgb2gray(observe), (110, 84), mode='constant') * 255)
    return processed_observe

def saveFile(reward_sum, timestamps):
    np.savetxt("history/spaceInvader_keras_DQN/spaceInvader_rewards.txt",reward_sum, fmt= '%d')

    currentTime = calendar.timegm(time.gmtime())
    timestamps.append(currentTime)
    np.savetxt("history/spaceInvader_keras_DQN/spaceInvader_timestamps.txt",timestamps, fmt= '%d')

def loadFile():
    Rewards = np.loadtxt("history/spaceInvader_keras_DQN/spaceInvader_rewards.txt", dtype=int)
    timestamps = np.loadtxt("history/spaceInvader_keras_DQN/spaceInvader_timestamps.txt", dtype=int)
    return Rewards.tolist(), timestamps.tolist()

def plotGraph(number_eps, rewards):
    plt.plot(number_eps, rewards, linestyle='--')
    ymin = min(rewards)
    ymax = max(rewards)

    plt.ylim(ymin - 10,ymax + 10)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.show()

    yhat = savgol_filter(rewards, 41, 2)

    plt.plot(number_eps,yhat)
    plt.show()

def trainModel(isResume, isRender):
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    # Deterministic-v4 version use 4 actions
    EPISODES = 100000
    resume = isResume
    render = isRender
    saveFreq = 500
    modelChkpntFreq = 5000
    env = gym.make('SpaceInvadersDeterministic-v4')
    agent = model_using_DQN(action_size=3, modelDir='history/spaceInvader_keras_DQN/', 
                            fileName='history/spaceInvader_keras_DQN/spaceInvader_dqn_weights.h5', summaryfolder = 'history/spaceInvader_keras_DQN/summary/spaceInvader_dqn', statesize=(110, 84, 4), resume=resume)

    scores, episodes, global_step = [], [], 0

    if resume is True:
        scores, timestamps = loadFile()
    else:
        scores, timestamps = [], []
    
    episode_number = len(scores)

    agent.avg_q_max, agent.avg_loss = 0, 0

    while episode_number < EPISODES:
        episode_number += 1
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 110, 84, 4))
        while not done:
            if render:
                env.render()
            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = get_action(history, agent)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 3
            else:
                real_action = 4

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 110, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # agent.avg_q_max += np.amax(
            #     agent.model.predict(np.float32(history / 255.))[0])

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, dead)
            # every some time interval, train model
            agent.train_replay()
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history
    

            # if done, plot the score over episodes
            if done:
                scores.append(score)

                # if global_step > agent.train_start:
                #     stats = [score, agent.avg_q_max / float(step), step,
                #              agent.avg_loss / float(step)]
                #     for i in range(len(stats)):
                #         agent.sess.run(agent.update_ops[i], feed_dict={
                #             agent.summary_placeholders[i]: float(stats[i])
                #         })
                #     summary_str = agent.sess.run(agent.summary_op)
                #     agent.summary_writer.add_summary(summary_str, episode_number + 1)
                #     print(summary_str,e)

                print("episode:", episode_number, "  score:", score, "  memory length:",
                      len(agent.memory),
                      "  global_step:", global_step)

                agent.avg_q_max, agent.avg_loss = 0, 0

                if episode_number % saveFreq == 0:
                    agent.save_model("history/spaceInvader_keras_DQN/spaceInvader_dqn_weights.h5")
                    saveFile(scores, timestamps)

                if episode_number % modelChkpntFreq == 0:
                    filename = 'spaceInvader_dqn_weights_' + str(episode_number) + '.h5'
                    agent.saveCheckpoint(filename)

def demoModel(filename):
    env = gym.make('SpaceInvadersDeterministic-v4')
    agent = model_using_DQN(action_size=3, modelDir = 'history/spaceInvader_keras_DQN/', 
                            fileName=filename, summaryfolder = 'history/spaceInvader_keras_DQN/summary/spaceInvader_dqn', resume =True, statesize=(84, 84, 4))
    
    episode_number = 0
    
    demo_episodes = 10
    while episode_number < demo_episodes:
        episode_number += 1
        
        done = False
        dead = False

        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            time.sleep(0.02)
            env.render()

            # get action for the current history and go one step in environment
            action = get_action(history, agent)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
                time.sleep(1)
            else:
                history = next_history

def plotRewards():
    reward_sum_array = loadFile()
    number_eps = np.arange(len(reward_sum_array))
    plotGraph(number_eps, reward_sum_array)

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"ht:r:R:g:d:",["help","train=","resume=","render=","graph=","demochkpt="])
except getopt.GetoptError:
    print('python3 spaceInvader_keras_DQN.py --train True --resume True --render False --graph True')
    sys.exit(2)

isGraph = False
demoChkpt = 0

for opt, arg in opts:
    if opt == '-h':
        print('python3 spaceInvader_keras_DQN.py --train True --resume True --render False')
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
    trainModel(isResume == 'True', isRender == 'True')
else:
    print('Starting from demo checkpoint : ', demoChkpt)
    
    if demoChkpt == 0:
        filename = 'history/spaceInvader_keras_DQN/spaceInvader_dqn_weights.h5'
    else:
        filename = 'history/spaceInvader_keras_DQN/spaceInvader_dqn_weights_' + str(demoChkpt) + '.h5'

    demoModel(filename)