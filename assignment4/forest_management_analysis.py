import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import mdp as mdp

def Forest_Experiments():
	import mdptoolbox, mdptoolbox.example
	for prob_size in [100, 2000]:
		print('POLICY ITERATION WITH FOREST MANAGEMENT')
		P, R = mdptoolbox.example.forest(S=prob_size)
		value_f = [0]*10
		policy = [0]*10
		iters = [0]*10
		time_array = [0]*10
		gamma_arr = [0] * 10
		for i in range(0,10):
			pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=(i+0.5)/10, max_iter=100000)
			pi.run()
			gamma_arr[i]=(i+0.5)/10
			value_f[i] = np.max(pi.V)
			policy[i] = pi.policy
			iters[i] = pi.iter
			time_array[i] = pi.time

		print('VALUE ITERATION WITH FOREST MANAGEMENT')
		P, R = mdptoolbox.example.forest(S=prob_size)
		val_value_f = [0]*10
		val_policy = [0]*10
		val_iters = [0]*10
		val_time_array = [0]*10
		val_gamma_arr = [0] * 10
		for i in range(0,10):
			pi = mdptoolbox.mdp.ValueIteration(P, R, discount=(i+0.5)/10, max_iter=100000)
			pi.run()
			val_gamma_arr[i]=(i+0.5)/10
			val_value_f[i] = np.max(pi.V)
			val_policy[i] = pi.policy
			val_iters[i] = pi.iter
			val_time_array[i] = pi.time

		print(val_value_f)
		print(value_f)

		cdict = {'Value Iteration': 'red', 'Policy Iteration': 'blue'}
		plt.plot(val_gamma_arr, val_time_array, c=cdict['Value Iteration'])
		plt.plot(gamma_arr, time_array, c=cdict['Policy Iteration'])
		plt.plot()
		plt.xlabel('Gammas')
		plt.title('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Execution Time Analysis')
		plt.ylabel('Execution Time (s)')
		plt.legend(cdict)
		plt.savefig('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Execution Time Analysis' + str('.png'))
		plt.clf()
	
		plt.plot(val_gamma_arr, val_value_f, c=cdict['Value Iteration'])
		plt.plot(gamma_arr,value_f, c=cdict['Policy Iteration'])
		plt.xlabel('Gammas')
		plt.ylabel('Average Rewards')
		plt.title('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Reward Analysis')
		plt.legend(cdict)
		plt.savefig('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Reward Analysis' + str('.png'))
		plt.clf()

		plt.plot(val_gamma_arr, val_iters, c=cdict['Value Iteration'])
		plt.plot(gamma_arr,iters, c=cdict['Policy Iteration'])
		plt.xlabel('Gammas')
		plt.ylabel('Iterations to Converge')
		plt.title('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Convergence Analysis')
		plt.legend(cdict)
		plt.savefig('State Size: '+ str(prob_size) +' Forest Management - Value Iteration - Convergence Analysis' + str('.png'))
		plt.clf()
	
	print('Q LEARNING WITH FOREST MANAGEMENT')
	iteration = 1000000
	for prob_size in [100, 2000]:
		prob = 0.1
		P, R = mdptoolbox.example.forest(S=prob_size,p=prob, r1=10, r2=1)
		value_f = []
		value_max = []
		policy = []
		iters = []
		time_array = []
		Q_table = []
		rew_array = []
		for epsilon in [0.05,0.15,0.25,0.5,0.75,0.95]:
			st = time.time()
			pi = mdp.QLearning(P,R,0.95, n_iter=iteration)
			end = time.time()
			pi.run(epsilon)
			rew_array.append(pi.reward_array)
			#value_f.append(np.mean(pi.V))
			#value_max.append(np.max(pi.V))
			policy.append(pi.policy)
			time_array.append(end-st)
			Q_table.append(pi.Q)

		plt.plot(range(0,1000000), rew_array[0],label='epsilon=0.05')
		plt.plot(range(0,1000000), rew_array[1],label='epsilon=0.15')
		plt.plot(range(0,1000000), rew_array[2],label='epsilon=0.25')
		plt.plot(range(0,1000000), rew_array[3],label='epsilon=0.50')
		plt.plot(range(0,1000000), rew_array[4],label='epsilon=0.75')
		plt.plot(range(0,1000000), rew_array[5],label='epsilon=0.95')
		plt.legend()
		plt.xlabel('Iterations')
		plt.grid()
		plt.title('Size: '+str(prob_size)+' With prob'+ str(prob) + 'Forest Management - Q Learning - Decaying Epsilon')
		plt.ylabel('Average Reward')
		plt.savefig('Size: '+str(prob_size)+' With prob'+ str(prob) +' Forest Management - Q Learning - Decaying Epsilon' + str('.png'))
		plt.clf()

		plt.subplot(1,6,1)
		plt.imshow(Q_table[0][:20,:])
		plt.title('Epsilon=0.05')

		plt.subplot(1,6,2)
		plt.title('Epsilon=0.15')
		plt.imshow(Q_table[1][:20,:])

		plt.subplot(1,6,3)
		plt.title('Epsilon=0.25')
		plt.imshow(Q_table[2][:20,:])

		plt.subplot(1,6,4)
		plt.title('Epsilon=0.50')
		plt.imshow(Q_table[3][:20,:])

		plt.subplot(1,6,5)
		plt.title('Epsilon=0.75')
		plt.imshow(Q_table[4][:20,:])

		plt.subplot(1,6,6)
		plt.title('Epsilon=0.95')
		plt.imshow(Q_table[5][:20,:])
		plt.colorbar()
		plt.savefig('Size: ' + str(prob_size) + 'SubPlot Forest Management - Q Learning - Decaying Epsilon' + str('.png'))
		plt.clf()

print('STARTING EXPERIMENTS')
#Frozen_Lake_Experiments()
Forest_Experiments()
#Taxi_Experiments()
print('END OF EXPERIMENTS')




