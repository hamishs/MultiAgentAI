import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt
import time

import random
from collections import deque

class cfg:
	# env
	n_agents = 10
	n_states = 6
	n_actions = 5 + n_agents

	buffer_size = 10
	max_len = 100

	# hyperparameters
	gamma = 0.99
	alpha = 0.04
	lr = 1e-3

	# model parameters
	channels = 10
	kernel_shape = 3
	pool_kernel = 3
	controller_feats = 32
	c_layers = 3

	# training params
	episodes = 200000
	train_steps = 2

class EpisodicBuffer:
	'''Buffer to store full episodes at a time.'''

	def __init__(self, buffer_size, max_len = 50):
		self.buffer_size = buffer_size
		self.max_len = max_len
		self.reset()

	def reset(self):
		self.buffer = deque(maxlen = self.buffer_size)

	def __len__(self):
		return len(self.buffer)

	def start_episode(self):
		self.buffer.append([])

	def update(self, s, a, r, d):
		n = len(self.buffer) - 1
		if len(self.buffer[n]) < self.max_len:
			self.buffer[n].append([s, a, r, d])

	def sample(self):
		idx = np.random.randint(0, len(self.buffer)-2) # exclude incomplete
		batch = self.buffer[idx] 
		return tuple(zip(*batch))

class Controller(nn.Module):

	def __init__(self, n_states, n_actions, n_agents, channels, kernel_size, pool_kernel, controller_feats, n_blocks):
		super().__init__()

		self.n_states = n_states
		self.n_agents = n_agents 
		self.n_actions = n_actions

		in_feats = channels * ((7 - kernel_size - pool_kernel) ** 2)
		self.controller_feats = controller_feats

		# encoder
		self.conv = nn.Conv2d(n_states, channels, kernel_size)
		self.pool = nn.AvgPool2d(pool_kernel)
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(in_feats, controller_feats)

		# controller blocks
		self.blocks = nn.ModuleList([nn.Linear(2*controller_feats, controller_feats) for _ in range(n_blocks)])
		self.linear_p = nn.Linear(controller_feats, n_actions)
		self.soft = nn.Softmax(dim = -1)

		# value network
		self.linear_v = nn.Linear(controller_feats * n_agents, 1)
		
	def encoder(self, s):
		# s : (batch, n_agents, n_states, 5, 5)
		n_agents = s.shape[1]

		# encode
		s = s.reshape(-1, self.n_states, 5, 5)
		s = self.conv(s)
		s = self.pool(s)
		s = self.flatten(s)
		s = F.relu(self.linear1(s))
		return s.reshape(-1, n_agents, self.controller_feats) # (batch, agents, feats)
	
	def policy(self, s):
		n_agents = s.shape[1]
		h = self.encoder(s)

		# blocks
		c = (h.sum(dim = 1, keepdim = True) - h) / (n_agents - 1)
		for block in self.blocks:
			x = torch.cat([h, c], dim = -1)
			h = block(x)
			c = (h.sum(dim = 1, keepdim = True) - h) / (n_agents - 1)
		
		return self.soft(self.linear_p(h))

	def value(self, s):
		h = self.encoder(s) # (batch, agents, feats)
		h = h.reshape(-1, self.n_agents * self.controller_feats)
		return self.linear_v(h)
	
	def forward(self, s):
		n_agents = s.shape[1]
		h = self.encoder(s)

		v = h.reshape(-1, self.n_agents * self.controller_feats)
		v = self.linear_v(v)

		# blocks
		c = (h.sum(dim = 1, keepdim = True) - h) / (n_agents - 1)
		for block in self.blocks:
			x = torch.cat([h, c], dim = -1)
			h = block(x)
			c = (h.sum(dim = 1, keepdim = True) - h) / (n_agents - 1)
		
		q = self.soft(self.linear_p(h))

		return q, v

class CommNet:

	def __init__(self, n_states, n_actions, gamma, n_agents,
				 model, buffer_size, max_len, alpha, lr, device):
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.n_agents = n_agents
		self.buffer = EpisodicBuffer(buffer_size, max_len)

		self.alpha = alpha

		self.device = device

		self.model = model.to(device)
		self.policy = self.model.policy
		self.value = self.model.value

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

		self.log_min = torch.FloatTensor([1e-10])

	def act(self, s, exploration = True):
		# s : (n_agents, n_states, 5, 5)
		s = s.float().to(self.device)
		q = self.policy(s[np.newaxis, ...])
		q = q.squeeze().detach().cpu().numpy()

		if exploration:
			return [int(np.random.choice(cfg.n_actions, p = q[i])) for i in range(self.n_agents)]
		else:
			return np.argmax(q, axis = -1).tolist()

	def train(self):
		s, a, r, d = self.buffer.sample()

		s = torch.from_numpy(np.array(s)).float().to(self.device) # (T, n_agents, n_states, 5, 5)
		a = torch.from_numpy(np.array(a)).long().to(self.device) # (T, n_agents)
		r = torch.from_numpy(np.array(r)).float().to(self.device) # (T, n_agents)
		d = torch.from_numpy(np.array(d)).float().to(self.device) # (T, n_agents)

		self.optimizer.zero_grad()
		loss = self.loss_fn(s, a, r)
		loss.backward()
		self.optimizer.step()

		return loss.cpu().item()

	def loss_fn(self, s, a, r):

		# future reward
		r = torch.cumsum(r.sum(-1), 0) # (T,)

		# feedforward
		q, v = self.model(s) # (T, n_agents, n_actions), (T, 1)

		# critic loss
		critic_loss = torch.mean(torch.square(r - v.squeeze(-1)))

		# actor loss
		q = torch.gather(q, -1, torch.unsqueeze(a, -1)).squeeze(-1) # (T, n_agents)
		q = torch.log(torch.maximum(q, self.log_min)).sum(-1) # (T,)
		with torch.no_grad():
			scale_ = r - v.squeeze(-1)
		actor_loss = torch.mean(q * scale_)

		return self.alpha * critic_loss - actor_loss
		
	def update_buffer(self, *args):
		self.buffer.update(*args)

	
model = Controller(cfg.n_states, cfg.n_actions, cfg.n_agents, cfg.channels,
				   cfg.kernel_shape, cfg.pool_kernel, cfg.controller_feats, cfg.c_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cmn = CommNet(cfg.n_states, cfg.n_actions, cfg.gamma, cfg.n_agents,
			  model, cfg.buffer_size, cfg.max_len, cfg.alpha, cfg.lr, device)
env = Combat(grid_shape=(15, 15), n_agents=cfg.n_agents, n_opponents=cfg.n_agents)

# training loop
verbose = 100
ep_rewards = []
losses = []
start_ = time.time()
for episode in range(cfg.episodes):
	s = env.reset()
	cmn.buffer.start_episode()
	s = torch.from_numpy(np.array(s).reshape(cfg.n_agents, cfg.n_states, 5, 5))
	d = [h == 0 for h in env.agent_health.values()]
	ep_reward = 0.0
	while not all(d):

		# act
		a = cmn.act(s)
		s_next, r, _, _ = env.step(a)
		d = [h == 0 for h in env.agent_health.values()]
		ep_reward += np.sum(r)
		s_next = torch.from_numpy(np.array(s_next).reshape(cfg.n_agents, cfg.n_states, 5, 5))
		cmn.update_buffer(s.numpy(), a, r, d)
		s = s_next
		ep_reward += np.sum(r)

	ep_rewards.append(ep_reward)

	if len(cmn.buffer) >= 3:
		for _ in range(cfg.train_steps):
			losses.append(cmn.train())
	
	if episode % verbose == 0:
		print('Episode {} Reward {:.4f} Loss {:.4f} in {:.4f} secs per ep'.format(
			episode, np.mean(ep_rewards[-cfg.train_steps * verbose:]),
			np.mean(losses[-cfg.train_steps * verbose:]),
			(time.time() - start_)/verbose))
		start_ = time.time()
