import sys, os
import numpy as np
sys.path.append('.')
from frozen_lake import FrozenLakeEnv
from value_iteration_agent import ValueIterationAgent
from policy_iteration_agent import PolicyIterationAgent
from ValueIteration import value_iteration
from utils import visualize_policy, visualize_value, visualize_env, evaluate_value_iteration, plot_policy_map, colors_lake, directions_lake, evaluate_policy
from constants import FL4x4
#from utils import *
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
        #for grid_size in [10, 15, 20, 25]:
        for grid_size in [10]:
            #Q-learning
            st = time.time()
            reward_array = []
            iter_array = []
            size_array = []
            chunks_array = []
            averages_array = []
            time_array = []
            Q_array = []
            custom_rewards = {b'F': -0.00004, b'H': -1, b'G': 1, b'S': -0.00004}  # (move, hole, goal, start)
            env = FrozenLakeEnv(desc=None, map_name=None, is_slippery=True, random_map_size=grid_size,
                                custom_rewards=custom_rewards, slip_prob=0.2)
            env = env.unwrapped
            desc = env.unwrapped.desc
            #for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
            for epsilon in [0.5, 0.75, 0.90]:
                Q = np.zeros((env.observation_space.n, env.action_space.n))
                rewards = []
                iters = []
                optimal = [0] * env.observation_space.n
                alpha = 0.85
                gamma = 0.95
                episodes = 5000
                for episode in range(episodes):
                    print(str(episode)+'====>'+str(grid_size)+'===>'+str(epsilon))
                    state = env.reset()
                    done = False
                    t_reward = 0
                    max_steps = 1000000
                    for i in range(max_steps):
                        #print(i)
                        if done:
                            break
                        current = state
                        if np.random.rand() < (epsilon):
                            action = np.argmax(Q[current, :])
                        else:
                            action = env.action_space.sample()

                        state, reward, done, info = env.step(action)
                        t_reward += reward
                        Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
                    epsilon = (1 - 2.71 ** (-episode / 1000))
                    rewards.append(t_reward)
                    iters.append(i)

                for k in range(env.observation_space.n):
                    optimal[k] = np.argmax(Q[k, :])

                reward_array.append(rewards)
                iter_array.append(iters)
                Q_array.append(Q)

                env.close()
                end = time.time()
                # print("time :",end-st)
                time_array.append(end - st)


                # Plot results
                def chunk_list(l, n):
                    for i in range(0, len(l), n):
                        yield l[i:i + n]


                size = int(episodes / 50)
                chunks = list(chunk_list(rewards, size))
                averages = [sum(chunk) / len(chunk) for chunk in chunks]
                size_array.append(size)
                chunks_array.append(chunks)
                averages_array.append(averages)

            print('Grid Size:' +str(grid_size))
            print(size)
            print(chunks)
            print(averages)
            print(size_array)
            print(chunks_array)
            print(averages_array)

            #plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsilon=0.05')
            #plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsilon=0.15')
            #plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.25')
            plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsilon=0.50')
            plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsilon=0.75')
            plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.95')
            plt.legend()
            plt.xlabel('Iterations')
            plt.grid()
            plt.title('Grid Size: ' + str(grid_size) + 'Frozen Lake - Q Learning - Constant Epsilon')
            plt.ylabel('Average Reward')
            plt.savefig('Grid Size: ' + str(grid_size) + ' Frozen Lake - Q Learning - Constant Epsilon' + str('.png'))
            plt.clf()

            plt.plot([0.5, 0.75, 0.95], time_array)
            plt.xlabel('Epsilon Values')
            plt.grid()
            plt.title('Grid Size: ' + str(grid_size) + 'Frozen Lake - Q Learning')
            plt.ylabel('Execution Time (s)')
            plt.savefig('Grid Size: ' + str(grid_size) + ' Frozen Lake - Q Learning vs Time' + str('.png'))
            plt.clf()

            """plt.subplot(1, 6, 1)
            plt.imshow(Q_array[0])
            plt.title('Epsilon=0.05')

            plt.subplot(1, 6, 2)
            plt.title('Epsilon=0.15')
            plt.imshow(Q_array[1])

            plt.subplot(1, 6, 3)
            plt.title('Epsilon=0.25')
            plt.imshow(Q_array[2])"""

            plt.subplot(1, 6, 4)
            plt.title('Epsilon=0.50')
            plt.imshow(Q_array[0])

            plt.subplot(1, 6, 5)
            plt.title('Epsilon=0.75')
            plt.imshow(Q_array[1])

            plt.subplot(1, 6, 6)
            plt.title('Epsilon=0.95')
            plt.imshow(Q_array[2])
            plt.colorbar()
            plt.savefig('Grid Size: ' + str(grid_size) + ' Frozen Lake - Epsilon vs Q_array' + str('.png'))
            plt.clf()

        #visualize_policy(agent.extract_policy, FL4x4, env.desc.shape, 'Optimal policy - Modified transition model')
        #visualize_value(agent.values, FL4x4, env.desc.shape, 'Value estimates - Modified transition model')
