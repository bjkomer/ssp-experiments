import gym
import time
import gridworlds
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from envs import create_env, get_max_dist

import argparse

parser = argparse.ArgumentParser('Train and RL agent on a gridworld task')

parser.add_argument('--n-steps', type=int, default=1000000, help='total timesteps to train for')
parser.add_argument('--n-demo-steps', type=int, default=1000, help='total timesteps to view demo for')
parser.add_argument('--movement-type', type=str, default='holonomic', choices=['holonomic', 'directional'])
parser.add_argument('--ssp-dim', type=int, default=256)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--n-sensors', type=int, default=0)
parser.add_argument('--continuous', type=int, default=1, choices=[1, 0])
parser.add_argument('--fname', type=str, default='')
parser.add_argument('--env-size', type=str, default='tiny', choices=['miniscule', 'tiny', 'small', 'medium', 'large'])
parser.add_argument('--seed', type=int, default=13, help='seed for the SSP axis vectors')
parser.add_argument('--curriculum', action='store_true', help='gradually increase goal distance during training')
parser.add_argument('--regular-coordinates', action='store_true', help='use 2D coordinates instead of SSP')
parser.add_argument('--demo-goal-distance', type=int, default=0, help='goal distance used for demo. 0 means any distance')
parser.add_argument('--train-goal-distance', type=int, default=0, help='goal distance to use during training')
parser.add_argument('--eval-freq', type=int, default=100000, help='how many steps between eval runs')
parser.add_argument('--n-eval-episodes', type=int, default=100, help='how many episodes for the evaluation')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

env = create_env(goal_distance=args.train_goal_distance, args=args)
eval_env = create_env(goal_distance=args.train_goal_distance, args=args)

if not os.path.exists('models'):
    os.makedirs('models')

# If no model name specified, generate based on parameters
if args.fname == '':
    fname = 'models/{}_{}dim_{}hs_{}sensors_{}seed_{}steps'.format(
        args.env_size, args.ssp_dim, args.hidden_size, args.n_sensors, args.seed, args.n_steps
    )
else:
    fname = args.fname

# load or train model
if os.path.exists(fname + '.zip'):
    model = PPO.load(fname)
else:
    # policy_kwargs = dict(layers=[args.ssp_dim])
    policy_kwargs = dict(net_arch=[args.hidden_size])
    model = PPO('MlpPolicy', env, verbose=args.verbose, policy_kwargs=policy_kwargs)

    n_evals = int(args.n_steps // args.eval_freq)
    steps_per_eval = int(args.n_steps // n_evals)
    # include additional eval at the start
    eval_data = np.zeros((n_evals + 1, 2))

    # for curriculum learning, start at the easiest setting
    if args.curriculum:
        cur_envs = []
        max_dist = get_max_dist(args.env_size)
        steps_per_dist = args.n_steps // (max_dist + 1)
        goal_distance = 1
        for i in range(1, max_dist):
            cur_envs.append(create_env(goal_distance=i, args=args))
        # env.close()
        # del env
        # env = create_env(goal_distance=goal_distance, args=args)
        # model.set_env(env)

    total_steps = 0
    for eval in range(n_evals):
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes)
        eval_data[eval, 0] = mean_reward
        eval_data[eval, 1] = std_reward
        print("Total Steps: {}".format(total_steps))
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print("")
        model.learn(total_timesteps=steps_per_eval)
        total_steps += steps_per_eval

        # potentially update curriculum if there is one
        if args.curriculum:
            if total_steps >= steps_per_dist * goal_distance:
                goal_distance += 1
                print("updating curriculum to distance: {}".format(goal_distance))
                # env.close()
                # del env
                # env = create_env(goal_distance=goal_distance, args=args)
                # model.set_env(env)
                model.set_env(cur_envs[min(goal_distance-1, max_dist-1)])

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes)
    eval_data[-1, 0] = mean_reward
    eval_data[-1, 1] = std_reward
    print("Total Steps: {}".format(total_steps))
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("")
    print("Saving data to {}".format(fname))

    np.savez(fname + '.npz', eval_data=eval_data)
    model.save(fname)


display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    print("No Display detected, skipping demo view")
else:
    env = create_env(goal_distance=args.demo_goal_distance, args=args)
    # demo model
    obs = env.reset()
    for i in range(args.n_demo_steps):
        action, _states = model.predict(obs, deterministic=True)
        # action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        time.sleep(0.001)

env.close()
