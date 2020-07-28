import gym
import time
import gridworlds
import os
import numpy as np
from envs import create_env, get_max_dist

import argparse

parser = argparse.ArgumentParser('Train an RL agent on a gridworld task')

parser.add_argument('--n-steps', type=int, default=1000000, help='total timesteps to train for')
parser.add_argument('--eval-freq', type=int, default=100000, help='how many steps between eval runs')
parser.add_argument('--n-eval-episodes', type=int, default=100, help='how many episodes for the evaluation')
parser.add_argument('--n-demo-steps', type=int, default=1000, help='total timesteps to view demo for')
parser.add_argument('--deterministic-demo', type=int, default=1, choices=[0, 1], help='demo with deterministic actions or not')
parser.add_argument('--movement-type', type=str, default='holonomic', choices=['holonomic', 'directional'])
parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'a2c', 'sac', 'td3'])
parser.add_argument('--ssp-dim', type=int, default=256)
parser.add_argument('--ssp-scaling', type=float, default=0.5)
# parser.add_argument('--hidden-size', type=int, default=256)
# parser.add_argument('--hidden-layers', type=int, default=1)
parser.add_argument('--hidden-size', type=int, default=2048)
parser.add_argument('--hidden-layers', type=int, default=2)
parser.add_argument('--n-sensors', type=int, default=0)
parser.add_argument('--continuous', type=int, default=1, choices=[1, 0])
parser.add_argument('--env-size', type=str, default='tiny', choices=['miniscule', 'tiny', 'small', 'medium', 'large'])
parser.add_argument('--seed', type=int, default=13, help='seed for the SSP axis vectors')
parser.add_argument('--curriculum', action='store_true', help='gradually increase goal distance during training')
parser.add_argument('--regular-coordinates', action='store_true', help='use 2D coordinates instead of SSP')
parser.add_argument('--st-ssp', action='store_true', help='use sub-toroid ssp')
parser.add_argument('--demo-goal-distance', type=int, default=0, help='goal distance used for demo. 0 means any distance')
parser.add_argument('--train-goal-distance', type=int, default=0, help='goal distance to use during training')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--use-open-env', action='store_true', help='if true, env will have no interior walls')
parser.add_argument('--discrete-actions', type=int, default=0,
                    help='if set, use this many discrete actions instead of continuous')
# parser.add_argument('--discrete-actions', type=int, default=4,
#                     help='if set, use this many discrete actions instead of continuous')

parser.add_argument('--pseudoreward-mag', type=float, default=5)
parser.add_argument('--pseudoreward-std', type=float, default=5)

parser.add_argument('--fname', type=str, default='')
parser.add_argument('--save-periodically', action='store_true')
parser.add_argument('--backend', type=str, default='tensorflow', choices=['tensorflow', 'pytorch'])
parser.add_argument('--max-steps', type=int, default=100, help='maximum steps per episode')
parser.add_argument('--fixed-episode-length', action='store_true')

args = parser.parse_args()


if args.backend == 'pytorch':
    from stable_baselines3 import PPO, A2C, SAC, TD3
    # from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.evaluation import evaluate_policy
elif args.backend == 'tensorflow':
    from stable_baselines import A2C, SAC, TD3
    from stable_baselines import PPO2 as PPO
    # from stable_baselines.ppo2 import MlpPolicy
    from stable_baselines.common.evaluation import evaluate_policy
else:
    raise NotImplementedError

if args.algo == 'ppo':
    Algo = PPO
elif args.algo == 'a2c':
    Algo = A2C
elif args.algo == 'sac':
    Algo = SAC
elif args.algo == 'td3':
    Algo = TD3
else:
    raise NotImplementedError


model = Algo.load(args.fname)

env = create_env(goal_distance=args.demo_goal_distance, args=args, max_steps=100)
# demo model
obs = env.reset()
print(obs)
print(type(obs))
print(obs.shape)
# env.close()
# assert False
for i in range(args.n_demo_steps):
    action, _states = model.predict(obs, deterministic=args.deterministic_demo)
    print(action)
    print(action.shape)
    # env.close()
    # assert False
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    time.sleep(0.0001)

env.close()
