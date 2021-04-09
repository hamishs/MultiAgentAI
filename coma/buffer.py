from collections import deque 
import torch

class EpisodicBuffer:

    def __init__(self, buffer_size, maxlen = 100):
        '''
        buffer_size : maximum length of deque
        maxlen : maximum length of episode to store
        '''
        self.buffer_size = buffer_size
        self.maxlen = maxlen
        self.state = deque(maxlen = buffer_size)
        self.action = deque(maxlen = buffer_size)
        self.reward = deque(maxlen = buffer_size)
        self.done = deque(maxlen = buffer_size)
    
    def reset(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.done.clear()
        self.bootstrapps = []
        self.targets = []
    
    def start_episode(self):
        self.state.append([])
        self.action.append([])
        self.reward.append([])
        self.done.append([])
    
    def update(self, s, a, r, d):
        n = len(self.state) - 1
        if len(self.state[n]) < self.maxlen:
            self.state[n].append(s)
            self.action[n].append(a)
            self.reward[n].append(r)
            self.done[n].append(d)
        
    def compute_bootstraps(self, critic, device):
        ''' Computes state estimates for each state in memory.'''
        self.bootstraps = []
        for i in range(len(self.state)):
            s = torch.tensor(self.state[i], dtype = torch.float32).to(device)
            a = torch.tensor(self.action[i], dtype = torch.int32).to(device)
            f_s = critic(s, a) # (ep_length, 1)
            self.bootstraps.append(f_s.squeeze().detach().cpu().tolist())
    
    def compute_lamda_return(self, r_disc, b, lamda, gamma):
        assert r_disc.shape == b.shape
        T = r_disc.shape[0]

        ys = []
        for t in range(1,T+1):
            x1 = (lamda ** (T - t)) * r_disc[t:].sum()

            x2 = 0
            for n in range(1, T-t+1):
                x2 += (gamma ** (-1-t)) * r_disc[t:t+n].sum()
                if t + n - 1 < T:
                    x2 += (gamma ** n) * b[t+n-1]  
            x2 *= 1.0 - lamda

            ys.append(x1 + x2)
        
        return ys
        
    def compute_targets(self, critic, device, gamma, lamda):
        ''' Computes lambda returns with a given lamda, gamma.'''

        self.targets = []
        self.compute_bootstraps(critic, device)
        
        for ep in range(len(self.state)):

            # compute discounted rewards
            r = np.array(self.reward[ep])
            discs = gamma ** np.arange(r.shape[0])
            r_disc = r * discs

            # get bootstrapped states
            b = np.array(self.bootstraps[ep])

            # compute targets
            ys = self.compute_lamda_return(r_disc, b, lamda, gamma)
            self.targets.append(ys)