import torch 
import numpy as np 
from coma.buffer import EpisodicBuffer

class COMA:

    def __init__(self, n_states, n_agents, n_actions, gamma, lamda, actor, critic,
                 critic_target, lr_actor, lr_critic, epsilon, buffer_size, maxlen, device):
        self.n_states = n_states
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.gamma = gamma
        self.lamda = lamda

        self.device = device
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.update_target()

        self.actor_optim = torch.optim.Adam(actor.parameters(), lr = lr_actor)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr = lr_critic)

        self.buffer = EpisodicBuffer(buffer_size, maxlen=maxlen)

        self.epsilon = epsilon
        self._t = 0
    
    def act(self, s, exploration = True):
        ''' s : (n_agents, n_states, 5, 5)'''

        # get action distribution from policy for each agent
        a, self.hidden_states = self.actor(s, self.hidden_states)
        a = a.detach().cpu().numpy() # (n_agents, n_actions)

        # get epsilon
        self._t += 1
        eps = self.epsilon(self._t) if callable(self.epsilon) else self.epsilon

        # sample actions for each agent
        return [int(np.random.choice(self.n_actions, p = a_ * (1.0 - eps) + eps/self.n_actions)) for a_ in a]
        
    def train(self):
        device = self.device

        self.compute_targets()
        states, actions, targets = self.buffer.state, self.buffer.action, self.buffer.targets 

        s = torch.vstack([torch.tensor(s_, dtype = torch.float32) for s_ in states]).to(device)
        s = s.reshape(-1, self.n_agents, self.n_states, 5, 5)
        a = torch.vstack([torch.tensor(a_, dtype = torch.int32) for a_ in actions]).to(device)
        y = torch.cat([torch.tensor(y_, dtype = torch.float32) for y_ in targets]).to(device)

        # critic loss on td targets
        self.critic_optim.zero_grad()
        critic_loss = torch.mean(torch.square(self.critic(s, a).squeeze() - y))
        critic_loss.backward()
        self.critic_optim.step()

        # actor loss
        self.actor_optim.zero_grad()
        actor_loss = 0.0
        for ep in range(len(states)):
            h = self.actor.get_hidden(self.n_agents)
            for s, u in zip(states[ep], actions[ep]):
                s = torch.tensor(s, dtype = torch.float32).to(device)
                s = s.reshape(self.n_agents, self.n_states, 5, 5)
                u = torch.tensor(u, dtype = torch.int32).to(device)
                base, pi, h = self.baseline(s, u, h)
                log_pi_u = torch.log(torch.gather(pi, 1, u.long().unsqueeze(1))).squeeze(1)
                actor_loss += -log_pi_u.dot(base)
        actor_loss.backward()
        self.actor_optim.step()

        return critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()

    def baseline(self, s, u, h):
        '''
        Computes the baseline for agents with joint action u,
        current state s and given actor and critic networks.
        s : (n_agents, n_states, 5, 5) current state
        u : (n_agents) joint actions taken
        h : (n_layers, n_agents, hidden) hidden state of actor
        '''
        with torch.no_grad():
            q = self.critic(s[np.newaxis, ...], u[np.newaxis, ...]).squeeze()
        pi, h_new = self.actor(s, h) # (n_agents, n_actions), (n_layers, T, hidden)

        # get counterfactual actions
        cf_actions = u.unsqueeze(0).unsqueeze(0).repeat(self.n_agents, self.n_actions, 1) # (n_agents, n_actions, n_agents)
        for i in range(self.n_agents):
            cf_actions[i,:,i] = torch.arange(cf_actions.shape[1], dtype = cf_actions.dtype)
        cf_actions = cf_actions.reshape(-1, self.n_agents)

        # compute cf baseline
        with torch.no_grad():
            q_cf = self.critic(s.unsqueeze(0).repeat(cf_actions.shape[0], 1, 1, 1, 1), cf_actions).squeeze()
        q_cf = q_cf.reshape(self.n_agents, self.n_actions).t()
        base = q - torch.matmul(pi, q_cf).diagonal()
        
        return base, pi, h_new
            
    def compute_targets(self):
        self.buffer.compute_targets(self.critic_target, self.device, self.gamma, self.lamda)
    
    def update_buffer(self, s, a, r, d):
        self.buffer.update(s, a, r, d)
    
    def start_episode(self):
        self.hidden_states = self.actor.get_hidden(self.n_agents)
        self.buffer.start_episode()
    
    def clear_buffer(self):
        self.buffer.reset()
    
    def update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())