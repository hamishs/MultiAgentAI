import argparse
import yaml

class Loader:
    def __init__(self, input_obj):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        input_dict = {k:v for k,v in input_dict.items() if v is not None}
        with open('config.yml', 'r') as f:
            custom_params = yaml.safe_load(f)[input_dict['config_name']]
        combined_dict = {**custom_params, **input_dict}
        self._generate_loader(combined_dict)
       
    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            setattr(self, key, val)

class train_parser:

	parser = argparse.ArgumentParser(description='MNMT Input Params, Training')

	###### required arguments ######
	parser.add_argument(
	    '--run_name', required=True,
	    type=str, help="Name of run"
	)
	parser.add_argument(
	    '--config_name', default='base',
	    type=str, help="Name of config params."
	)

	###### action arguments ######
	parser.add_argument(
		'--wandb', action='store_true',
		help="Will record run in weights and biases."
	)
	parser.add_argument(
		'--save', action='store_true',
		help="Save results and model."
	)

	###### optional arguments (overwrite config) ######
	parser.add_argument(
	    '--grid_shape', default=None,
	    type=int, help="Size of grid."
	)
	parser.add_argument(
	    '--n_agents', default=None,
	    type=int, help="Number of agents (same for opponents)."
	)
	parser.add_argument(
	    '--gamma', default=None,
	    type=float, help="Discount rate."
	)
	parser.add_argument(
	    '--lamda', default=None,
	    type=float, help="Lambda return constant."
	)
	parser.add_argument(
	    '--epsilon', default=None,
	    type=float, help="Max clip in advantage."
	)
	parser.add_argument(
	    '--v_weight', default=None,
	    type=float, help="Weight of critic loss to actor loss."
	)
	parser.add_argument(
	    '--e_weight', default=0.01,
	    type=float, help="Strength of entropy regularisation."
	)
	parser.add_argument(
	    '--buffer_size', default=None,
	    type=int, help="Max episodes to store in buffer."
	)
	parser.add_argument(
	    '--channels', default=None,
	    type=int, help="Number of channels of CNN."
	)
	parser.add_argument(
	    '--kernel_size', default=None,
	    type=int, help="Kernel size of CNN."
	)
	parser.add_argument(
	    '--hidden_size', default=None,
	    type=int, help="Hidden size of MLP."
	)
	parser.add_argument(
	    '--lr_actor', default=None,
	    type=float, help="Learning rate for actor network."
	)
	parser.add_argument(
	    '--lr_critic', default=None,
	    type=float, help="Learning rate for critic network."
	)
	parser.add_argument(
	    '--episodes', default=None,
	    type=int, help="Number of episodes to train for."
	)
	parser.add_argument(
	    '--train_freq', default=None,
	    type=int, help="Frequency of episodes with training."
	)
	parser.add_argument(
	    '--train_steps', default=None,
	    type=int, help="Number of gradient steps when training."
	)
	parser.add_argument(
	    '--verbose', default=None,
	    type=int, help="Frequency of episodes to print stats."
	)
	parser.add_argument(
		'--activation', default=None,
		type=str, help="Activation function either relu or tanh."
	)
	parser.add_argument(
		'--num_cnn_layers', default=None,
		type=str, help="Activation function either relu or tanh."
	)
	parser.add_argument(
		'--actor_layers', default=None,
		type=str, help="Activation function either relu or tanh."
	)
	parser.add_argument(
		'--critic_layers', default=None,
		type=str, help="Activation function either relu or tanh."
	)
	

	@staticmethod
	def parse_args():
		return train_parser.parser.parse_args()



