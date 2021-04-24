import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Model(nn.Module):

    def __init__(self, n_agents, n_actions, d_model, kernel_size, activation=nn.ReLU,
        num_cnn_layers=3, num_policy_layers=3, num_value_layers=3):
        super().__init__() 

        self.embedding = nn.Embedding(2 * n_agents + 2, d_model)

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([nn.Conv2d(d_model, d_model, kernel_size), activation()])
        self.cnn = nn.Sequential(*cnn_layers)
        
        policy_layers = [nn.Linear(2*d_model, d_model), activation()]
        for _ in range(num_policy_layers-2):
            policy_layers.extend([nn.Linear(d_model, d_model), activation()])
        policy_layers.extend([nn.Linear(d_model, n_actions), nn.Softmax(dim = -1)])
        self.mlp_policy = nn.Sequential(*policy_layers)
        
        value_layers = []
        for _ in range(num_value_layers-1):
            value_layers.extend([nn.Linear(d_model, d_model), activation()])
        value_layers.extend([nn.Linear(d_model, 1)])
        self.mlp_value = nn.Sequential(*value_layers)

    def forward(self, s):
        '''
        Base CNN
        s : (batch, length, width) current board position 
        '''

        # embedded board state
        s = self.embedding(s) # (batch, length, width, d_model)

        # cnn
        s = self.cnn(s.permute(0, 3, 1, 2)) # (batch, d_model, ...)
        return F.max_pool2d(s, s.size()[2:]).squeeze(-1).squeeze(-1) # (batch, d_model)
    
    def value(self, s):
        '''
        Value network.
        i : (batch,)
        s : (batch, length, width) current board position 
        '''
        s = self.forward(s) # (batch, d_model)
        return self.mlp_value(s) # (batch, 1)
    
    def policy(self, s, agent_id):
        '''
        Policy network
        s : (batch, length, width) current board position 
        agent_id : (batch,) int id of the agent to generate actions for.
        '''
        s = self.forward(s) # (batch, d_model)
        agent_id = self.embedding(agent_id) # (batch, d_model)
        x = torch.cat([s, agent_id], dim = -1) # (batch, 2 * d_model)
        return self.mlp_policy(x) # (batch, n_actions)