import torch

class TrajectoryBuffer:
	''' Buffer to store trajectories in order and store advantages.'''
	
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.reset()
		
	def reset(self):
		self.state = []
		self.action = []
		self.reward = []
		self.done = []
		self.state_next = []
		self.probs = []
		self.advs = []
		self.tds = torch.FloatTensor([])
		self.n = 0

	def __len__(self):
		return len(self.state)
	
	def update(self, s, a, r, d, s_next, pi):
		self.state.append(s)
		self.action.append(a)
		self.reward.append(r)
		self.done.append(d)
		self.state_next.append(s_next)
		self.probs.append(pi)
		
		if self.n < self.buffer_size:
			self.n += 1
		else:
			self.state.pop(0)
			self.action.pop(0)
			self.reward.pop(0)
			self.done.pop(0)
			self.state_next.pop(0)
			self.probs.pop(0)
			self.advs.pop(0)
			self.tds = self.tds[1:]
	
	@property
	def data(self):
		''' Return memory as arrays.'''
		s = torch.FloatTensor(self.state)
		a = torch.LongTensor(self.action)
		r = torch.FloatTensor(self.reward)
		d = torch.FloatTensor(self.done)
		s_next = torch.FloatTensor(self.state_next)
		pi = torch.FloatTensor(self.probs)
		advs = torch.FloatTensor(self.advs)
		return s, a, r, d, s_next, pi, advs, self.tds