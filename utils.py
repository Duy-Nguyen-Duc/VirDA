import os
import random
import string
from yacs.config import CfgNode as CN


def create_random_exp_tag(directory: str):
    """
    Create a random experiment tag for logging purposes.
    """
    tag = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    if os.path.exists(os.path.join(directory, tag)):
        return create_random_exp_tag(directory)
    return tag

def setup(cfg: CN):
    current_dir = os.path.join(os.getcwd(), "runs")
    os.makedirs(current_dir, exist_ok=True)
    exp_code = create_random_exp_tag(current_dir)
    exp_save_dir = os.path.join(current_dir, exp_code)
    os.makedirs(exp_save_dir, exist_ok=True)
    with open(os.path.join(exp_save_dir, "config.txt"), "w") as f:
        f.write(cfg.dump())
    return exp_save_dir