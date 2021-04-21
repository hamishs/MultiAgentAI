import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from copy import deepcopy

from buffer import TrajectoryBuffer

class MAPPO:
    ''' Multi-agent Proximal Policy Optimisation. Shared actor parameters.
    Omniscient critic.'''

    def __init__(self, board_shape, n_actions, n_agents, gamma, lmda, epsilon, v_weight,
                 e_weight, buffer_size, model, lr, device, max_to_keep=0):

        self.board_l, self.board_w = board_shape
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.gamma = gamma
        self.device = device

        self.lmda = lmda
        self.epsilon = epsilon
        self.v_weight = v_weight
        self.e_weight = e_weight

        self.buffer = TrajectoryBuffer(buffer_size)

        # model contains both policy and critic networks
        self.model = model
        self.policy = self.model.policy 
        self.value = self.model.value 
        self.opt = torch.optim.Adam(self.model.parameters(), lr = lr)

        self.max_to_keep = max_to_keep
        self.prev_policies = []

    def act(self, s, return_prob = False, exploration = True):
        # s : (board_l, board_w) current board state
        s = s.unsqueeze(0).repeat(self.n_agents, 1, 1) # (n_agents, board_l, board_w)
        agent_ids = torch.arange(self.n_agents) + 2 # (n_agents)
        s, agent_ids = self.to_device((s, agent_ids))

        s1, s2 = s[:3], s[3:]
        ids1, ids2 = agent_ids[:3], agent_ids[3:]

        with torch.no_grad():
            p = self.policy(s, agent_ids)
        p = p.cpu().numpy()

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
        self.opt.zero_grad()
        loss = self.loss_fn(s, a, r, d, s_next, pi_old, advs, tds)
        loss.backward()
        self.opt.step()

        return loss.detach().cpu().item()

    def loss_fn(self, s, a, r, d, s_next, pi_old, advs, tds):

        # get action distribution for each agent in each state
        b = s.size()[0]
        s = s.reshape(-1, self.board_l, self.board_w).unsqueeze(1) # (batch, ...)
        s = s.repeat(1, self.n_agents, 1, 1).reshape(-1, self.board_l, self.board_w) # (batch * agents, ...)
        agent_id = torch.arange(self.n_agents).repeat(b).to(self.device) # (batch * agents,)
        pi_new = self.policy(s, agent_id).reshape(-1, self.n_agents, self.n_actions) # (batch, agents, actions)
        
        # actor loss
        ratio = torch.exp(torch.log(torch.gather(pi_new, 2, a).squeeze()) - torch.log(pi_old))
        actor_loss = torch.minimum(ratio * advs, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs) 

        # entropy
        e = torch.distributions.Categorical(pi_new).entropy()

        # critic loss
        v = self.value(s).reshape(-1, self.n_agents) # (batch, n_agents)
        v_loss = 0.5 * torch.square((v - tds).squeeze())

        # compute loss and update
        loss = (-actor_loss + self.v_weight * v_loss - self.e_weight * e)
        return loss.mean()

    def update_buffer(self, s, a, r, d, s_next, pi):
        self.buffer.update(s, a, r, d, s_next, pi)

    def compute_advantages(self, l):
        ''' Compute advantage estimates for the previous l steps of the buffer.'''

        s = torch.IntTensor(self.buffer.state[-l:]).reshape(-1, self.board_l, self.board_w) # (l, board_l, board_w)
        r = torch.FloatTensor(self.buffer.reward[-l:]) # (l, n_agents)
        s_next = torch.IntTensor(self.buffer.state_next[-l:]).reshape(-1, self.board_l, self.board_w) # (l, board_l, board_w)
        d = torch.FloatTensor(self.buffer.done[-l:]) # (l, 1)

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
    
    def update_prev_policies(self):
        if len(self.prev_policies) + 1 >= self.max_to_keep:
            self.prev_policies.pop(0)
        
        self.prev_policies.append(deepcopy(self.model))