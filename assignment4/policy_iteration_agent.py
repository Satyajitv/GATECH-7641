from time import time

import numpy as np


class PolicyIterationAgent:

	def __init__(self, env, gamma):
		self.max_iterations = 10000
		self.gamma = gamma
		self.num_states = env.observation_space.n
		self.num_actions = env.action_space.n
		self.state_prob = env.P

		self.values = np.zeros(env.observation_space.n)
		self.policy = np.zeros(env.observation_space.n)

	def compute_policy_v(self, env, policy, gamma):
		v = np.zeros(env.nS)
		eps = 1e-5
		while True:
			prev_v = np.copy(v)
			for s in range(env.nS):
				policy_a = policy[s]
				v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
			if (np.sum((np.fabs(prev_v - v))) <= eps):
				break
		return v

	def extract_policy(self, env, v, gamma):
		policy = np.zeros(env.nS)
		for s in range(env.nS):
			q_sa = np.zeros(env.nA)
			for a in range(env.nA):
				q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
			policy[s] = np.argmax(q_sa)
		return policy

	def policy_iteration(self, env, gamma):
		k = 0
		policy = np.random.choice(env.nA, size=(env.nS))
		max_iters = 200000
		desc = env.unwrapped.desc
		for i in range(max_iters):
			old_policy_v = self.compute_policy_v(env, policy, gamma)
			new_policy = self.extract_policy(env, old_policy_v, gamma)
			if (np.all(policy == new_policy)):
				k = i + 1
				break
			policy = new_policy
			self.policy = new_policy
			print(i)
		return self.policy, k

	def choose_action(self, observation):
		return self.policy[observation]
