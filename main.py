import sys
import os
sys.path.append(os.path.abspath("./tdmpc2"))
from tdmpc2.train import train

if __name__ == '__main__':

  train("config.yaml")
