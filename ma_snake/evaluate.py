import torch
import numpy as np 

from utils import action_wrapper, get_rewards

class Evaluate:
	""" Class to evaluate performance of agents."""

	def __init__(self, env, cfg):
		self.env = env 
		self.n_agents = cfg.n_agents
		self.board_l, self.board_w = cfg.board_l, cfg.board_w
		self.n_actions = cfg.n_actions

	def one_vs_one(self, agent1, agent2, episodes=1):
		""" Play agent 1 against agent 2 for some episodes.
		agents are a function taking the raw state and returning the raw action (for all agents)"""
		results = []
		for episode in range(episodes):
			s = self.env.reset()
			d = False
			while not d:
				a1, a2 = agent1(s)[:self.n_agents], agent2(s)[self.n_agents:]
				a = a1 + a2
				s, r, d, _, _ = self.env.step(action_wrapper(a))
			
			# check winner
			w = get_rewards(s)[0][0]
			results.append(w)
		
		# return win rate of agent 1
		return np.mean(results)*0.5 + 0.5

	def vs_random(self, mappo, episodes=1):
		""" Evaluate agent against random policy. """

		# define agent functions
		def agent1(s):
			s = torch.IntTensor(s).reshape(self.board_l, self.board_w)
			return mappo.act(s, exploration = False)
		agent2 = lambda s: np.random.randint(self.n_actions, size = (6,)).tolist()

		# run evaluation
		return one_vs_one(agent1, agent2, episodes=episodes)

	def elo_tourne(self, agents, ratings=None, episodes=10, K=32):
		""" Run an elo tournement to rank the given agents."""

		# baseline is random agent
		random_agent = lambda s: np.random.randint(self.n_actions, size = (6,)).tolist()
		agents = [random_agent] + agents

		# initiate ratings at 1000 if not given
		if ratings is not None:
			ratings = np.ones(len(agents)) * 1000
		else:
			ratings = np.array([1000] + ratings)

		# run the tournament
		for i, p1 in enumerate(agents):
			for j, p2 in enumerate(agents):
				if i != j:
					# current ratings and expected scores
					r1, r2 = ratings[i], ratings[j]
					q1, q2 = 10. ** (r1 / 400.), 10. ** (r2 / 400.)
					es1, es2 = q1/(q1 + q2), q2/(q1 + q2)

					# get win rate in 1v1 matches
					w1 = self.one_vs_one(p1, p2, episodes=episodes)
					w2 = 1.0 - w1

					# update elos
					ratings[i] += K * (w1 - es1)
					ratings[j] += K * (w2 - es2)

		# rescale random to 1000
		ratings *= 1000 / ratings[0]
		return ratings

