import sys, os
sys.path.append(os.path.abspath(".."))
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from .tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from envs import make_env, make_vec_env
from common.logger import Logger


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

@parse_cfg
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	# assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	set_seed(cfg.seed)
	print(colored('Work dir:', 'green', attrs=['bold']), cfg.work_dir)

	cfg.bin_size = (cfg.vmax - cfg.vmin) / cfg.num_bins

	trainer_cls = OnlineTrainer
	torch.autograd.set_detect_anomaly(True)
	trainer = trainer_cls(
		cfg=cfg,
		env=make_vec_env(cfg),
		eval_env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')
