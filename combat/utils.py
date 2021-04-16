import argparse
import yaml

class Loader:
    def __init__(self, input_obj):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        with open('config.yml', 'r') as f:
            custom_params = yaml.safe_load(f)[input_dict['config_name']]
        combined_dict = {**input_dict, **custom_params}
        self._generate_loader(combined_dict)
       
    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            setattr(self, key, val)

class train_parser:

	parser = argparse.ArgumentParser(description='MNMT Input Params, Training')

	parser.add_argument(
	    '--run_name', required=True,
	    type=str, help="Name of run"
	)
	parser.add_argument(
	    '--config_name', default='base',
	    type=str, help="Name of config params."
	)

	@staticmethod
	def parse_args():
		return train_parser.parser.parse_args()



