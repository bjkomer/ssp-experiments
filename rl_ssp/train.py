import gym
import time
import gridworlds
import os

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from envs import create_env

import argparse

parser = argparse.ArgumentParser('Train and RL agent on a gridworld task')

parser.add_argument('--n-steps', type=int, default=100000, help='total timesteps to train for')
parser.add_argument('--n-demo-steps', type=int, default=1000, help='total timesteps to view demo for')
parser.add_argument('--movement-type', type=str, default='holonomic', choices=['holonomic', 'directional'])
parser.add_argument('--ssp-dim', type=int, default=256)
parser.add_argument('--n-sensors', type=int, default=0)
parser.add_argument('--continuous', type=int, default=1, choices=[1, 0])
parser.add_argument('--fname', type=str, default='')
parser.add_argument('--env-size', type=str, default='tiny', choices=['miniscule', 'tiny', 'small', 'medium', 'large'])
parser.add_argument('--seed', type=int, default=13, help='seed for the SSP axis vectors')

args = parser.parse_args()

env = create_env(args)

if not os.path.exists('models'):
    os.makedirs('models')

# If no model name specified, generate based on parameters
if args.fname == '':
    fname = 'models/{}_{}dim_{}sensors_{}seed_{}steps'.format(
        args.env_size, args.ssp_dim, args.n_sensors, args.seed, args.n_steps
    )
else:
    fname = args.fname

# load or train model
if os.path.exists(fname + '.zip'):
    model = PPO.load(fname)
else:

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.n_steps)

    model.save(fname)


# demo model
obs = env.reset()
for i in range(args.n_demo_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    time.sleep(0.001)

env.close()
