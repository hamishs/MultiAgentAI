from utils import Loader, train_parser
from ppo.train import train

if __name__ == '__main__':
	
	args = train_parser.parse_args()
	cfg = Loader(args)

	train(cfg)