from utils import Loader, train_parser
from ppo.train import train

if __name__ == '__main__':
	
	args = train_parser.parse_args()
	cfg = Loader(args)

	cfg.verbose = 2
	cfg.episodes = 10
	train(cfg)