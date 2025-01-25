import re
from types import SimpleNamespace
import yaml

def parse_cfg(func):
  def wrapper(cfg):
    with open(cfg, 'r') as f:
      cfg = yaml.safe_load(f)
    # Logic
    for k in cfg.keys():
      try:
        v = cfg[k]
        if v == None:
          v = True
      except:
        pass
        
    # Algebraic expressions
    for k in cfg.keys():
      try:
        v = cfg[k]
        if isinstance(v, str):
          match = re.match(r"(\d+)([+\-*/])(\d+)", v)
          if match:
            cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
            if isinstance(cfg[k], float) and cfg[k].is_integer():
              cfg[k] = int(cfg[k])
      except:
        pass
    func(SimpleNamespace(**cfg))
  
  return wrapper
