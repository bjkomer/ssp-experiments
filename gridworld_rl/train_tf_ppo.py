# Train with curriculum learning using the DeepRL code
import argparse
from gridworlds.envs import GridWorldEnv
from env_utils import get_env
from gridworlds.getch import getch
import numpy as np
import sys
from run_ppo import run_ppo
import os

parser = argparse.ArgumentParser("Train an agent to navigate to goals in a 2D world")

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--fname', type=str, default='tf_ppo')

args = parser.parse_args()

env = get_env()

if not os.path.exists(args.fname):
    os.makedirs(args.fname)

run_ppo(env, args.fname)
