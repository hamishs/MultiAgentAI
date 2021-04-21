import torch
import torch.nn as nn 
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, n_states, n_actions, channels, kernel_size, hidden_size,
        num_cnn_layers = 3, num_layers=3, activation=nn.ReLU):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        c_layers = [nn.Conv2d(n_states, channels, kernel_size, padding=(kernel_size - 1)//2), activation()]
        for _ in range(num_cnn_layers - 1):
            c_layers.extend([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2), activation()])
        self.cnn = nn.Sequential(*c_layers)

        mlp_layers = [nn.Linear(channels, hidden_size), activation()]
        for _ in range(num_layers - 2):
            mlp_layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        self.mlp = nn.Sequential(*mlp_layers)
        self.final = nn.Linear(hidden_size, n_actions)

    def forward(self, s, h = None):
        '''
        s : (batch, n_states, 5, 5)
        h : (n_layers, batch, hidden_size)
        '''

        # encoder
        s = self.cnn(s) # (batch, channels, ...,)
        s = F.max_pool2d(s, s.size()[2:]).squeeze(-1).squeeze(-1) # (batch, channels)

        # mlp (batch, hidden_size)
        s = self.mlp(s)

        # output
        s = F.softmax(self.final(s), -1) # (batch, n_actions)
        return s

class Critic(nn.Module):

    def __init__(self, n_states, n_agents, n_actions, channels, kernel_size, hidden_size,
        num_cnn_layers = 3, num_layers=2, activation=nn.ReLU()):
        super().__init__()

        self.n_states = n_states 
        self.n_agents = n_agents 
        self.n_actions = n_actions
        self.channels = channels

        c_layers = [nn.Conv2d(n_states, channels, kernel_size, padding=(kernel_size - 1)//2), activation()]
        for _ in range(num_cnn_layers - 1):
            c_layers.extend([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2), activation()])
        self.cnn = nn.Sequential(*c_layers)

        mlp_layers = [nn.Linear(channels * n_agents, hidden_size), activation()]
        for _ in range(num_layers - 2):
            mlp_layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        self.mlp = nn.Sequential(*mlp_layers)
        self.final = nn.Linear(hidden_size, 1)

    def forward(self, s):
        '''
        s : (batch, n_agents, n_states, 5, 5)
        '''

        # cnn for state
        s = s.reshape(-1, self.n_states, 5, 5)
        s = self.cnn(s) # (n_agents * batch, channels, ...)
        s = F.max_pool2d(s, s.size()[2:]) # (n_agents * batch, channels)
        s = s.reshape(-1, self.n_agents, self.channels).reshape(-1, self.n_agents*self.channels)

        # mlp
        s = self.mlp(s) # (batch, hidden_size)
        s = self.final(s)  # (batch, 1)

        return s

class CriticAgentInvariant(nn.Module):
    """ Critic model that is invariant to the number of agents. """

    def __init__(self, n_states, n_actions, channels, kernel_size, hidden_size,
        num_cnn_layers = 3, num_layers=2, activation=nn.ReLU()):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.channels = channels

        c_layers = [nn.Conv2d(n_states, channels, kernel_size, padding=(kernel_size - 1)//2), activation()]
        for _ in range(num_cnn_layers - 1):
            c_layers.extend([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2), activation()])
        self.cnn = nn.Sequential(*c_layers)

        mlp_layers = [nn.Linear(channels, hidden_size), activation()]
        for _ in range(num_layers - 2):
            mlp_layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        self.mlp = nn.Sequential(*mlp_layers)
        self.final = nn.Linear(hidden_size, 1)

    def forward(self, s):
        '''
        s : (batch, n_agents, n_states, 5, 5)
        '''

        n_agents = s.size(1)

        # cnn for state
        s = s.reshape(-1, self.n_states, 5, 5)
        s = self.cnn(s) # (n_agents * batch, channels, ...)
        s = F.max_pool2d(s, s.size()[2:]) # (n_agents * batch, channels)
        s = s.reshape(-1, n_agents, self.channels) # (batch, n_agents, channels)
        s = s.max(1)[0] # (batch, channels)

        # mlp
        s = self.mlp(s) # (batch, hidden_size)
        s = self.final(s)  # (batch, 1)

        return s