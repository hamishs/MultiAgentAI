import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Actor(nn.Module):

    def __init__(self, n_states, n_actions, channels,
                 kernel_size, n_layers, hidden_size):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(n_states, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU())
        self.gru = nn.GRU(input_size = channels,
                          hidden_size = hidden_size,
                          num_layers = n_layers)
        self.final = nn.Linear(hidden_size, n_actions)
    
    def forward(self, s, h = None):
        '''
        s : (batch, n_states, 5, 5)
        h : (n_layers, batch, hidden_size)
        '''

        # encoder
        s = self.cnn(s) # (batch, channels, ...,)
        s = F.max_pool2d(s, s.size()[2:]).squeeze(-1).squeeze(-1) # (batch, channels)

        # gru (1, batch, hidden_size) (n_layers, batch, hidden_size)
        s, h = self.gru(s.unsqueeze(0), h)

        # output
        s = F.softmax(self.final(s.squeeze(0)), -1) # (batch, n_actions)
        return s, h
  
    def get_hidden(self, batch):
        return torch.zeros(self.n_layers, batch, self.hidden_size)

class Critic(nn.Module):

    def __init__(self, n_states, n_agents, n_actions, channels, kernel_size):
        super().__init__()

        self.n_states = n_states 
        self.n_agents = n_agents 
        self.n_actions = n_actions
        self.channels = channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_states, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1)//2),
            nn.ReLU())
        
        self.emb = nn.Embedding(n_actions, channels)

        self.mlp = nn.Sequential(
            nn.Linear(2 * channels * n_agents, channels * n_agents),
            nn.ReLU(),
            nn.Linear(channels * n_agents, channels * n_agents // 2),
            nn.ReLU(),
            nn.Linear(channels * n_agents // 2, 1))
    
    def forward(self, s, a):
        '''
        s : (batch, n_agents, n_states, 5, 5)
        a : (batch, n_agents,)
        '''

        # cnn for state
        s = s.reshape(-1, self.n_states, 5, 5)
        s = self.cnn(s) # (n_agents * batch, channels, ...)
        s = F.max_pool2d(s, s.size()[2:]) # (n_agents * batch, channels)
        s = s.reshape(-1, self.n_agents, self.channels).reshape(-1, self.n_agents*self.channels)

        # action embedding
        a = self.emb(a) # (batch, n_agents, channels)
        a = a.reshape(-1, self.n_agents*self.channels)

        # mlp
        x = torch.cat([s, a], dim = 1) # (batch, 2 * n_agents * channels)
        x = self.mlp(x)  # (batch, 1)

        return x
