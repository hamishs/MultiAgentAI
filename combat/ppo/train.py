# -*- coding: utf-8 -*-
from ppo.ppo import MAPPO
from ppo.models import Actor, Critic
from env.combat import Combat

import torch
import numpy as np 
import pandas as pd
import os

def train(cfg):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# make env and agents
	env = Combat(grid_shape=(cfg.grid_shape, cfg.grid_shape), n_agents=cfg.n_agents, n_opponents=cfg.n_agents)
	critic = Critic(cfg.n_states, cfg.n_agents, cfg.n_actions, cfg.channels, cfg.kernel_size).to(device)
	actor = Actor(cfg.n_states, cfg.n_actions, cfg.channels, cfg.kernel_size, cfg.hidden_size).to(device)
	mappo = MAPPO(cfg.n_states, cfg.n_actions, cfg.n_agents, cfg.gamma, cfg.lamda,
				  cfg.epsilon, cfg.v_weight, cfg.e_weight, cfg.buffer_size,
				  actor, critic, cfg.lr_actor, cfg.lr_critic, device)

	# train
	ep_rewards = []
	losses = []
	win = []
	for episode in range(cfg.episodes):

		s = env.reset()
		done = False
		ep_length = 0
		ep_reward = 0.0
		while not done:
			a, pi = mappo.act(torch.FloatTensor(s).reshape(cfg.n_agents, cfg.n_states, 5, 5), return_prob = True)
			s_next, r, d, _ = env.step(a)

			d_agent = [v == 0 for _, v in env.agent_health.items()]
			d_opp = [v == 0 for _, v in env.opp_health.items()]

			if all(d_opp) or all(d_agent) or all(d):
				done = True

			ep_reward += np.sum(r)
			mappo.update_buffer(s, a, r, d_agent, s_next, pi)
			s = s_next
			ep_length += 1

		# compute game result
		if all(d_opp) and ((not all(d)) or (not all(d_agent))):
			win.append(1.0) # win
		elif (not all(d_opp)) and (not all(d_agent)) and all(d):
			win.append(0.5) # draw
		elif all(d_opp) and all(d_agent) and all(d):
			win.append(0.5) # draw 
		elif (not all (d_opp)) and all(d_agent):
			win.append(0.0) # loss
		else:
			print(d_opp, d_agent, d)
			raise ValueError

		mappo.compute_advantages(ep_length)
		ep_rewards.append(ep_reward)
		
		if episode % cfg.train_freq == 0:
			for _ in range(cfg.train_steps):
				loss = mappo.train()
				losses.append(loss)
		
		if episode % cfg.verbose == 0:
			print('Episode {} Reward {:.4f}  Win rate {:.2f} Loss {:.4f}'.format(episode,                                                
				np.mean(ep_rewards[-cfg.verbose:]),
				100*np.mean(win[-cfg.verbose:]),
				np.mean(losses[-cfg.verbose*cfg.train_steps:])))

	# save model and results
	os.mkdir(cfg.run_name)
	pd.DataFrame({'reward' : ep_rewards, 'win_rate' : win}).to_csv(cfg.run_name+'/results.csv', index=False)
	torch.save({'actor_state_dict':actor.state_dict(),
		'critic_state_dict':critic.state_dict()},
		cfg.run_name+'/checkpoint')

