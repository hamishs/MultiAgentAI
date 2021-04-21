from ma_snake.env.chooseenv import make
from ma_snake.buffer import TranjectoryBuffer
from ma_snake.ppo import MAPPO
from ma_snake.util import action_wrapper, get_rewards, evaluate
from ma_snake.models import Model

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

def train(cfg):

	if cfg.wandb:
		wandb.init(project='ma_snake', entity='hamishs',
			config = {k:v for k,v in cfg.__dict__.items() if isinstance(v, (float, int, str))})
		config = wandb.config

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	env = make('snakes_3v3')
	model = Model(cfg.n_agents, cfg.n_actions, cfg.d_model, cfg.kernel_size).to(device)
	mappo = MAPPO((cfg.board_l, cfg.board_w), cfg.n_actions, 2*cfg.n_agents, cfg.gamma,
	              cfg.lmda, cfg.epsilon, cfg.v_weight, cfg.e_weight, cfg.buffer_size,
	              model, cfg.lr, device)

	if cfg.wandb:
		wandb.watch(model)

	max_snakes, test_rates, losses = [], [], []
	for episode in range(cfg.episodes):
	    ep_length = 0
	    s = env.reset()
	    max_snake = 0
	    d = False
	    while not d:

	        a, pi = mappo.act(torch.IntTensor(s).reshape(cfg.board_l, cfg.board_w), return_prob = True)
	        s_next, r, d, _, _ = env.step(action_wrapper(a))
	        r, snake_length = get_rewards(s)
	        max_snake = max([snake_length, max_snake])

	        ep_reward += np.sum(r)
	        mappo.update_buffer(s, a, r, [d], s_next, pi)
	        s = s_next
	        ep_length += 1
	    
	    mappo.compute_advantages(ep_length)
	    ep_rewards.append(ep_reward)
	    max_snakes.append(max_snake)

	    if episode % cfg.train_freq == 0:
	        for _ in range(cfg.train_steps):
	            loss = mappo.train()
	            losses.append(loss)
	    
	    if episode % cfg.verbose == 0:
	    	_mean_snake = np.mean(max_snakes[-cfg.verbose:])
	    	_mean_loss = np.mean(losses[-cfg.verbose*cfg.train_steps:])
	        print('Episode {} Max snake {:.4f} Loss {:.4f}'.format(episode,                                                
	            _mean_snake,
	            _mean_loss))
	    
	    if cfg.wandb:
	    	wandb.log({'max_snake' : max_snakes[-1],
	    		'loss' : np.mean(losses[-cfg.train_steps:])})
	    
	    if episode % cfg.test_freq == 0:
	        test_rate = evaluate(cfg.test_episodes)*0.5 + 0.5
	        test_rates.append(test_rate)
	        print('Test win rate {:.4f}'.format(test_rate))
	        wandb.log({'test_rate':test_rate})

	# save model and results
	if cfg.save:
		os.mkdir(cfg.run_name)
		pd.DataFrame({'reward' : ep_rewards, 'win_rate' : win}).to_csv(cfg.run_name+'/results.csv', index=False)
		torch.save({'actor_state_dict':actor.state_dict(),
			'critic_state_dict':critic.state_dict()},
			cfg.run_name+'/checkpoint')
	if cfg.wandb:
		wandb.finish()


if __name__ == '__main__':
	from utils import Loader, train_parser
	args = train_parser.parse_args()
	cfg = Loader(args)
	train(cfg)