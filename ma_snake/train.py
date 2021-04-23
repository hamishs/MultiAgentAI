# -*- coding: utf-8 -*-
from env.chooseenv import make
from ppo import MAPPO
from utils import action_wrapper, get_rewards
from models import Model
from evaluate import Evaluate

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

def train(cfg):

	if cfg.wandb:
		wandb.init(project='ma_snake', entity='hamishs',
			config = {k:v for k,v in cfg.__dict__.items() if isinstance(v, (float, int, str))})
		config = wandb.config

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	env = make('snakes_3v3')
	activation = nn.Tanh if cfg.activation == 'tanh' else nn.ReLU
	model = Model(cfg.n_agents, cfg.n_actions, cfg.d_model,
		cfg.kernel_size, activation=activation).to(device)
	mappo = MAPPO((cfg.board_l, cfg.board_w), cfg.n_actions,
		2*cfg.n_agents, cfg.gamma, cfg.lmda, cfg.epsilon,
		cfg.v_weight, cfg.e_weight, cfg.buffer_size, model, cfg.lr, device,
		max_to_keep=cfg.max_to_keep)
	evaluator = Evaluate(env, cfg)

	if cfg.wandb:
		wandb.watch(model)

	max_snakes, test_rates, losses = [], [], []
	for episode in range(cfg.episodes):
		ep_length = 0
		s = env.reset()
		max_snake = 0
		d = False
		while not d:

			a, pi = mappo.act(torch.IntTensor(s).reshape(cfg.board_l, cfg.board_w), return_prob = True)
			s_next, r, d, _, _ = env.step(action_wrapper(a))
			r, snake_length = get_rewards(s)
			max_snake = max([snake_length, max_snake])

			mappo.update_buffer(s, a, r, [d], s_next, pi)
			s = s_next
			ep_length += 1
		
		mappo.compute_advantages(ep_length)
		max_snakes.append(max_snake)

		if episode % cfg.train_freq == 0:
			for _ in range(cfg.train_steps):
				loss = mappo.train()
				losses.append(loss)
		
		if episode % cfg.verbose == 0:
			_mean_snake = np.mean(max_snakes[-cfg.verbose:])
			_mean_loss = np.mean(losses[-cfg.verbose*cfg.train_steps:])
			print('Episode {} Max snake {:.4f} Loss {:.4f}'.format(episode,                                                
				_mean_snake,
				_mean_loss))
		
		if cfg.wandb:
			wandb.log({'max_snake' : max_snakes[-1],
				'loss' : np.mean(losses[-cfg.train_steps:])})

		if episode % cfg.record_model == 0:
			mappo.update_prev_policies()
		
		if episode % cfg.test_freq == 0 and episode > 0:
			elo_ratings = mappo.update_elo_ratings(evaluator, episodes=cfg.test_episodes, K=32)
			print('Elo ratings:', elo_ratings)
			print('Current elo {:.4f}'.format(elo_ratings[-1]))
			if cfg.wandb:
				wandb.log({'elo':elo_ratings[-1]})

	# save model and results
	if cfg.save:
		os.mkdir(cfg.run_name)
		torch.save({'model_state_dict':model.state_dict()},
			cfg.run_name+'/checkpoint')
	if cfg.wandb:
		wandb.finish()


if __name__ == '__main__':
	from utils import Loader, train_parser
	args = train_parser.parse_args()
	cfg = Loader(args)
	train(cfg)