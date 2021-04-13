import torch
import numpy as np

def action_wrapper(joint_action):
	'''
	Converst lits of actions to env action.
	:param joint_action:
	:return: wrapped joint action: one-hot
	'''
	joint_action_ = []
	for a in range(env.n_player):
		action_a = joint_action[a]
		each = [0] * env.action_dim
		each[action_a] = 1
		action_one_hot = [[each]]
		joint_action_.append([action_one_hot[0][0]])
	return joint_action_

def get_rewards(s):
	''' Updates the reward for each agent if their team has the longest snake.'''

	s = np.array(s).squeeze()
	snake_lengths = [(s == i).sum() for i in range(2, cfg.n_agents * 2 + 2)]
	team_1, team_2 = max(snake_lengths[:cfg.n_agents]), max(snake_lengths[cfg.n_agents:])
	max_len = max([team_1, team_2])

	if team_1 > team_2:
		return [1.0] * cfg.n_agents + [-1.0] * cfg.n_agents, max_len
	elif team_1 < team_2:
		return [-1.0] * cfg.n_agents + [1.0] * cfg.n_agents, max_len
	else:
		return [0.0] * 2 * cfg.n_agents, max_len

def evaluate(episodes):
	''' Tests the agent against a random policy.'''
	results = []
	for episode in range(episodes):
		s = env.reset()
		d = False
		while not d:
			a1 = mappo.act(torch.IntTensor(s).reshape(cfg.board_l, cfg.board_w), exploration = False)[:3]
			a2 = np.random.randint(cfg.n_actions, size = (3,)).tolist()
			a = a1 + a2
			s, r, d, _, _ = env.step(action_wrapper(a))
		
		# check winner
		w = get_rewards(s)[0][0]
		results.append(w)
	
	# return win rate
	return np.mean(results)