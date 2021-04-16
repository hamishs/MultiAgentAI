import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from ppo.buffer import TrajectoryBuffer
from ppo.models import Actor, Critic

class PPO:
	''' Proximal Policy Optimisation for single agent RL.'''

	def __init__(self, n_states, n_actions, gamma, lmda, epsilon, v_weight,
		e_weight, buffer_size, policy, value, lr_policy, lr_value, device):
		
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.device = device

		self.lmda = lmda
		self.epsilon = epsilon
		self.v_weight = v_weight
		self.e_weight = e_weight

		self.buffer = TrajectoryBuffer(buffer_size)

		# initiate networks and optimisers
		self.policy = policy
		self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)

		self.value = value
		self.opt_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)

	def act(self, s, return_prob = False):
		# s : (1, n_states)
		s = s.to(self.device)
		p = self.policy(s).squeeze().detach().cpu().numpy()
		a = int(np.random.choice(self.n_actions, p = p))
		if return_prob:
			return a, p[a]
		else:
			return a

	def train(self):
		''' Train the algorithm on the current buffer.'''
		s, a, r, d, s_next, pi_old, advs, tds = self.buffer.data
		advs = advs.squeeze()

		# loss and gradients
		self.opt_policy.zero_grad()
		self.opt_value.zero_grad()
		loss = self.loss_fn( s, a, r, d, s_next, pi_old, advs, tds)
		loss.backward()
		self.opt_policy.step()
		self.opt_value.step()	

		return loss.detach().cpu().item()

	def loss_fn(self, s, a, r, d, s_next, pi_old, advs, tds):

		# actor loss
		pi_new = self.policy(s)
		ratio = torch.exp(torch.log(torch.gather(pi_new, 1, a)).squeeze() - torch.log(pi_old))
		actor_loss = torch.minimum(ratio * advs, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs) 

		# entropy
		e = torch.distributions.Categorical(pi_new).entropy()
		
		# critic loss
		v_loss = 0.5 * torch.square((self.value(s) - tds).squeeze())
		
		# compute loss and update
		loss = (-actor_loss + self.v_weight * v_loss - self.e_weight * e)
		return loss.mean()

	def update_buffer(self, s, a, r, d, s_next, pi):
		self.buffer.update(s, a, r, d, s_next, pi)
		
	def compute_advantages(self, l):
		''' Compute advantage estimates for the previous l steps of the buffer.'''
		s = torch.FloatTensor(self.buffer.state[-l:]) # (l, n_states)
		r = torch.FloatTensor(self.buffer.reward[-l:]) # (l, 1)
		s_next = torch.FloatTensor(self.buffer.state_next[-l:]) # (l, n_states)
		d = torch.FloatTensor(self.buffer.done[-l:]) # (l, 1)

		device = self.device
		s, r, s_next, d = s.to(device), r.to(device), s_next.to(device), d.to(device)

		with torch.no_grad():
			td_target = r + self.gamma * self.value(s_next) * (1 - d)
			delta = (td_target - self.value(s)).squeeze() # (l,)
		
		# calculate advantage
		advantages = []
		adv = 0.0
		for d in delta.cpu().numpy()[::-1]:
			adv = self.gamma * self.lmda * adv + d
			advantages.append([adv])
		advantages.reverse() # (l, 1)
		
		# update tds
		if self.buffer.tds.shape == torch.Size([1, 0]):
			self.buffer.tds = td_target
		else:
			self.buffer.tds = torch.cat([self.buffer.tds, td_target], dim=0)
		self.buffer.advs += advantages

class MAPPO:
    ''' Multi-agent Proximal Policy Optimisation. Shared actor parameters.
    Omniscient critic.'''

    def __init__(self, n_states, n_actions, n_agents, gamma, lmda, epsilon, v_weight,
                 e_weight, buffer_size, policy, value, lr_policy, lr_value, device):

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.gamma = gamma
        self.device = device

        self.lmda = lmda
        self.epsilon = epsilon
        self.v_weight = v_weight
        self.e_weight = e_weight

        self.buffer = TrajectoryBuffer(buffer_size)

        # initiate networks and optimisers
        self.policy = policy
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)

        self.value = value
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)

    def act(self, s, return_prob = False, exploration = True):
        # s : (n_agents, n_states, ...)
        s = s.to(self.device)
        p = self.policy(s).squeeze().detach().cpu().numpy()
        actions, prob = [], []
        for i in range(self.n_agents):
            if exploration:			
                a = int(np.random.choice(self.n_actions, p = p[i]))
            else:
                a = int(np.argmax(p[i]))
            actions.append(a)
            prob.append(p[i,a])

        if return_prob:
            return actions, prob
        else:
            return actions

    def train(self):
        ''' Train the algorithm on the current buffer.'''
        s, a, r, d, s_next, pi_old, advs, tds = self.to_device(self.buffer.data)
        a = a.unsqueeze(2)

        # loss and gradients
        self.opt_policy.zero_grad()
        self.opt_value.zero_grad()
        loss = self.loss_fn( s, a, r, d, s_next, pi_old, advs, tds)
        loss.backward()
        self.opt_policy.step()
        self.opt_value.step()	

        return loss.detach().cpu().item()

    def loss_fn(self, s, a, r, d, s_next, pi_old, advs, tds):

        # actor loss
        pi_new = self.policy(s.reshape(-1, self.n_states, 5, 5)).reshape(-1, self.n_agents, self.n_actions)
        ratio = torch.exp(torch.log(torch.gather(pi_new, 2, a).squeeze()) - torch.log(pi_old))
        actor_loss = torch.minimum(ratio * advs, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs) 

        # entropy
        e = torch.distributions.Categorical(pi_new).entropy()

        # critic loss
        v_loss = 0.5 * torch.square((self.value(s) - tds).squeeze())

        # compute loss and update
        loss = (-actor_loss + self.v_weight * v_loss - self.e_weight * e)
        return loss.mean()

    def update_buffer(self, s, a, r, d, s_next, pi):
        self.buffer.update(s, a, r, d, s_next, pi)

    def compute_advantages(self, l):
        ''' Compute advantage estimates for the previous l steps of the buffer.'''

        s = torch.FloatTensor(self.buffer.state[-l:]).reshape(-1, self.n_agents, self.n_states, 5, 5) # (l, n_agents, *state)
        r = torch.FloatTensor(self.buffer.reward[-l:]) # (l, n_agents)
        s_next = torch.FloatTensor(self.buffer.state_next[-l:]) # (l, n_agents, *state)
        d = torch.FloatTensor(self.buffer.done[-l:]) # (l, n_agents)

        s, r, s_next, d = self.to_device((s, r, s_next, d))

        with torch.no_grad():
            td_target = r + self.gamma * self.value(s_next) * (1 - d) # (l, n_agents)
            delta = (td_target - self.value(s)) # (l, n_agents)

        # calculate advantage
        advantages = []
        adv = 0.0
        for d in delta.cpu().numpy()[::-1]:
            adv = self.gamma * self.lmda * adv + d
            advantages.append(adv)
        advantages.reverse() # (l, n_agents)

        # update tds
        if self.buffer.tds.shape == torch.Size([1, 0]):
            self.buffer.tds = td_target
        else:
            self.buffer.tds = torch.cat([self.buffer.tds, td_target], dim=0)
            self.buffer.advs += advantages
    
    def to_device(self, tensors):
        return (tensor.to(self.device) for tensor in tensors)





