import torch
import numpy as np
import argparse
import yaml

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

class Loader:
    def __init__(self, input_obj):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        input_dict = {k:v for k,v in input_dict.items() if v is not None}
        with open('config.yml', 'r') as f:
            custom_params = yaml.safe_load(f)[input_dict['config_name']]
        combined_dict = {**custom_params, **input_dict}
        self._generate_loader(combined_dict)
       
    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            setattr(self, key, val)

class train_parser:

	parser = argparse.ArgumentParser(description='MA-Snake training params.')

	###### required arguments ######
	parser.add_argument(
	    '--run_name', required=True,
	    type=str, help="Name of run"
	)
	parser.add_argument(
	    '--config_name', default='base',
	    type=str, help="Name of config params."
	)

	###### action arguments ######
	parser.add_argument(
		'--wandb', action='store_true',
		help="Will record run in weights and biases."
	)
	parser.add_argument(
		'--save', action='store_true',
		help="Save results and model."
	)

	###### optional arguments (overwrite config) ######
	parser.add_argument(
	    '--board_l', default=None,
	    type=int, help="Length of board."
	)
	parser.add_argument(
	    '--board_w', default=None,
	    type=int, help="Width of board."
	)
	parser.add_argument(
	    '--n_agents', default=None,
	    type=int, help="Number of snakes."
	)
	parser.add_argument(
	    '--d_model', default=None,
	    type=int, help="Model inner dimension."
	)
	parser.add_argument(
	    '--kernel_size', default=None,
	    type=int, help="Kernel size of CNN."
	)
	parser.add_argument(
	    '--gamma', default=None,
	    type=float, help="Discount rate."
	)
	parser.add_argument(
	    '--lamda', default=None,
	    type=float, help="Lambda return constant."
	)
	parser.add_argument(
	    '--epsilon', default=None,
	    type=float, help="Max clip in advantage."
	)
	parser.add_argument(
	    '--v_weight', default=None,
	    type=float, help="Weight of critic loss to actor loss."
	)
	parser.add_argument(
	    '--e_weight', default=None,
	    type=float, help="Strength of entropy regularisation."
	)
	parser.add_argument(
	    '--buffer_size', default=None,
	    type=int, help="Max episodes to store in buffer."
	)
	parser.add_argument(
	    '--lr', default=None,
	    type=float, help="Learning rate for model."
	)
	parser.add_argument(
	    '--episodes', default=None,
	    type=int, help="Number of episodes to train for."
	)
	parser.add_argument(
	    '--train_freq', default=None,
	    type=int, help="Frequency of episodes with training."
	)
	parser.add_argument(
	    '--train_steps', default=None,
	    type=int, help="Number of gradient steps when training."
	)
	parser.add_argument(
	    '--test_freq', default=None,
	    type=int, help="How often to do test runs."
	)
	parser.add_argument(
	    '--test_episodes', default=None,
	    type=int, help="Number of test runs."
	)
	parser.add_argument(
	    '--verbose', default=None,
	    type=int, help="Frequency of episodes to print stats."
	)

	@staticmethod
	def parse_args():
		return train_parser.parser.parse_args()



