import random
from collections import deque

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