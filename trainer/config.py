import pprint

class ModelConfig(object):

	def __init__(self, train_transform=None):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 128
		self.batch_size_cpu = 64	
		self.num_workers = 4
		self.epochs = 50
		self.dropout_value = 0.15
		self.train_transform = train_transform

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)
		
	def week_11_args(self):
		self.batch_size_cuda = 512
