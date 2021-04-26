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
    print("test")
    #Generating different grids for comparisions
    for size in [4]:
        rewards = {b'F':-0.00004, b'H': -1, b'G':1, b'S':-0.00004} #(move, hole, goal, start)
        env = FrozenLakeEnv(desc=None, map_name=None, is_slippery=True, random_map_size=size, custom_rewards=rewards, slip_prob=0.2)
        env = env.unwrapped
        gamma = 1.0
        theta = 0.0001
        desc = env.unwrapped.desc

        """agent = ValueIterationAgent(env, gamma)
        all_rewards = []
        vals, k = agent.value_iteration(theta)
        policy = agent.extract_policy()

        print("Agent Policy: ", agent.policy)
        print("Values: ", agent.values)
        print(len(agent.policy))
        print(directions_lake())
        evaluate_value_iteration(env, agent)
        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + ' (Value Iteration) ' + 'Gamma: ' + str(gamma),
            agent.policy.reshape(20, 20), desc, colors_lake(), directions_lake())"""

        best_vals = [0] * 10
        time_array = [0] * 10
        gamma_arr = [0] * 10
        iters = [0] * 10
        list_scores = [0] * 10
        for i in range(6, 10):
            st = time.time()
            #best_value, k = value_iteration(env, gamma=(i + 0.5) / 10)
            agent = ValueIterationAgent(env, gamma=1.0)
            vals, k = agent.value_iteration(theta)
            policy = agent.extract_policy()
            end = time.time()
            #policy_score = evaluate_policy(env, policy)
            #policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000, agent=agent, type='val')
            #policy_score = evaluate_value_iteration(env, agent)
            #gamma = (i + 0.5) / 10
            plot = plot_policy_map(
                'Grid Size: '+ str(size) +' Frozen Lake Value (Value Iteration) ' + 'Gamma: ' + str(gamma),
                agent.policy.reshape(size, size), desc, colors_lake(), directions_lake())
            #gamma_arr[i] = (i + 0.5) / 10
            iters[i] = k
            best_vals[i] = vals
            #list_scores[i] = np.mean(policy_score)
            time_array[i] = end - st

        print('Values:')
        print('best_vals: '+str(best_vals))
        print('time_array: '+str(time_array))
        print('iters: '+str(iters))
        #print('list_scores: '+str(list_scores))


        pol_time_array = [0] * 10
        pol_gamma_arr = [0] * 10
        pol_iters = [0] * 10
        pol_list_scores = [0] * 10
        for i in range(6, 10):
            st = time.time()
            agent = PolicyIterationAgent(env, gamma=(i + 0.5) / 10)
            best_policy, k = agent.policy_iteration(env, gamma=(i + 0.5) / 10)
            end = time.time()
            print(gamma)
            #scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / 10, agent=None, type='pol')
            #scores = evaluate_value_iteration(env, agent)
            plot = plot_policy_map(
                'Grid Size: '+ str(size) +' Frozen Lake Map (Policy Iteration) ' + 'Gamma: ' + str(gamma),
                agent.policy.reshape(size, size), desc, colors_lake(), directions_lake())
            pol_gamma_arr[i] = (i + 0.5) / 10
            #pol_list_scores[i] = np.mean(scores)
            pol_iters[i] = k
            pol_time_array[i] = end - st

        print('Policy:')
        print('pol_time_array: '+str(pol_time_array))
        print('pol_iters: '+str(pol_iters))
        #print('pol_list_scores: '+str(pol_list_scores))

        cdict = {'Value Iteration': 'red', 'Policy Iteration': 'blue'}
        plt.plot(gamma_arr, time_array, c=cdict['Value Iteration'])
        plt.plot(gamma_arr, pol_time_array, c=cdict['Policy Iteration'])
        plt.xlabel('Gammas')
        plt.title('Frozen Lake'+'Grid Size: ' +str(size) +' - Value vs Policy Iteration - Execution Time Analysis')
        plt.ylabel('Execution Time (s)')
        plt.grid()
        plt.legend(cdict)
        plt.savefig('Grid Size: '+str(size)+' Execution Time Analysis' + str('.png'))
        plt.clf()

        """plt.plot(gamma_arr, list_scores, c=cdict['Value Iteration'])
        plt.plot(gamma_arr, pol_list_scores, c=cdict['Policy Iteration'])
        plt.xlabel('Gammas')
        plt.ylabel('Average Rewards')
        plt.title('Frozen Lake'+'Grid Size: ' +str(size) +' - Value vs Policy Iteration - Reward Analysis')
        plt.grid()
        plt.legend(cdict)
        plt.savefig('Grid Size: '+str(size)+' Reward Analysis' + str('.png'))
        plt.clf()"""

        plt.plot(gamma_arr, iters, c=cdict['Value Iteration'])
        plt.plot(gamma_arr, pol_iters, c=cdict['Policy Iteration'])
        plt.xlabel('Gammas')
        plt.ylabel('Iterations to Converge')
        plt.title('Frozen Lake'+'Grid Size: ' +str(size) +' - Value vs Policy Iteration - Convergence Analysis')
        plt.grid()
        plt.legend(cdict)
        plt.savefig('Grid Size: '+str(size)+' Convergence Analysis' + str('.png'))
        plt.clf()

        print("complete")

        """plt.plot(gamma_arr, time_array)
        plt.xlabel('Gammas')
        plt.title('Frozen Lake - Value Iteration - Execution Time Analysis')
        plt.ylabel('Execution Time (s)')
        plt.grid()
        plt.show()

        plt.plot(gamma_arr, list_scores)
        plt.xlabel('Gammas')
        plt.ylabel('Average Rewards')
        plt.title('Frozen Lake - Value Iteration - Reward Analysis')
        plt.grid()
        plt.show()

        # plot_curves(gamma_arr, list_scores, )

        plt.plot(gamma_arr, iters)
        plt.xlabel('Gammas')
        plt.ylabel('Iterations to Converge')
        plt.title('Frozen Lake - Value Iteration - Convergence Analysis')
        plt.grid()
        plt.show()

        plt.plot(gamma_arr, best_vals)
        plt.xlabel('Gammas')
        plt.ylabel('Optimal Value')
        plt.title('Frozen Lake - Value Iteration - Best Value Analysis')
        plt.grid()
        plt.show()

        print("completed")"""
        #Run policy iteration
        """pi_policy, k = policy_iteration(env, gamma)
        plot = plot_policy_map('Frozen Lake Policy Map Iteration ' + 'Gamma: ' + str(gamma),
                            pi_policy.reshape(20, 20), desc, colors_lake(), directions_lake())"""


        #Q-learning
        """st = time.time()
        reward_array = []
        iter_array = []
        size_array = []
        chunks_array = []
        averages_array = []
        time_array = []
        Q_array = []
        custom_rewards = {b'F': -0.00004, b'H': -1, b'G': 1, b'S': -0.00004}  # (move, hole, goal, start)
        for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
            Q = np.zeros((env.observation_space.n, env.action_space.n))
            rewards = []
            iters = []
            optimal = [0] * env.observation_space.n
            alpha = 0.85
            gamma = 0.95
            episodes = 30000
            env = FrozenLakeEnv(desc=None, map_name=None, is_slippery=True, random_map_size=4, custom_rewards=custom_rewards, slip_prob=0.2)
            env = env.unwrapped
            desc = env.unwrapped.desc
            for episode in range(episodes):
                state = env.reset()
                done = False
                t_reward = 0
                max_steps = 1000000
                for i in range(max_steps):
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

        plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsilon=0.05')
        plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsilon=0.15')
        plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.25')
        plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsilon=0.50')
        plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4], label='epsilon=0.75')
        plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsilon=0.95')
        plt.legend()
        plt.xlabel('Iterations')
        plt.grid()
        plt.title('Frozen Lake - Q Learning - Constant Epsilon')
        plt.ylabel('Average Reward')
        plt.show()

        plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], time_array)
        plt.xlabel('Epsilon Values')
        plt.grid()
        plt.title('Frozen Lake - Q Learning')
        plt.ylabel('Execution Time (s)')
        plt.show()

        plt.subplot(1, 6, 1)
        plt.imshow(Q_array[0])
        plt.title('Epsilon=0.05')

        plt.subplot(1, 6, 2)
        plt.title('Epsilon=0.15')
        plt.imshow(Q_array[1])

        plt.subplot(1, 6, 3)
        plt.title('Epsilon=0.25')
        plt.imshow(Q_array[2])

        plt.subplot(1, 6, 4)
        plt.title('Epsilon=0.50')
        plt.imshow(Q_array[3])

        plt.subplot(1, 6, 5)
        plt.title('Epsilon=0.75')
        plt.imshow(Q_array[4])

        plt.subplot(1, 6, 6)
        plt.title('Epsilon=0.95')
        plt.imshow(Q_array[5])
        plt.colorbar()

        plt.show()"""

        #visualize_policy(agent.extract_policy, FL4x4, env.desc.shape, 'Optimal policy - Modified transition model')
        #visualize_value(agent.values, FL4x4, env.desc.shape, 'Value estimates - Modified transition model')
