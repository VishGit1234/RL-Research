import sys
import os
sys.path.append(os.path.abspath("./tdmpc2"))
from tdmpc2.inference import inference

if __name__ == '__main__':
  inference("config.yaml")
