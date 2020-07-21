import gym
import time
import gridworlds
import os
import numpy as np
from envs import create_env, get_max_dist

import argparse

parser = argparse.ArgumentParser('Train and RL agent on a gridworld task')

parser.add_argument('--n-steps', type=int, default=1000000, help='total timesteps to train for')
parser.add_argument('--eval-freq', type=int, default=100000, help='how many steps between eval runs')
parser.add_argument('--n-eval-episodes', type=int, default=100, help='how many episodes for the evaluation')
parser.add_argument('--n-demo-steps', type=int, default=1000, help='total timesteps to view demo for')
parser.add_argument('--deterministic-demo', type=int, default=1, choices=[0, 1], help='demo with deterministic actions or not')
parser.add_argument('--movement-type', type=str, default='holonomic', choices=['holonomic', 'directional'])
parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'a2c', 'sac', 'td3'])
parser.add_argument('--ssp-dim', type=int, default=256)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--hidden-layers', type=int, default=1)
# parser.add_argument('--hidden-size', type=int, default=2048)
# parser.add_argument('--hidden-layers', type=int, default=2)
parser.add_argument('--n-sensors', type=int, default=0)
parser.add_argument('--continuous', type=int, default=1, choices=[1, 0])
parser.add_argument('--env-size', type=str, default='tiny', choices=['miniscule', 'tiny', 'small', 'medium', 'large'])
parser.add_argument('--seed', type=int, default=13, help='seed for the SSP axis vectors')
parser.add_argument('--curriculum', action='store_true', help='gradually increase goal distance during training')
parser.add_argument('--regular-coordinates', action='store_true', help='use 2D coordinates instead of SSP')
parser.add_argument('--demo-goal-distance', type=int, default=0, help='goal distance used for demo. 0 means any distance')
parser.add_argument('--train-goal-distance', type=int, default=0, help='goal distance to use during training')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--use-open-env', action='store_true', help='if true, env will have no interior walls')
parser.add_argument('--discrete-actions', type=int, default=0,
                    help='if set, use this many discrete actions instead of continuous')
# parser.add_argument('--discrete-actions', type=int, default=4,
#                     help='if set, use this many discrete actions instead of continuous')

parser.add_argument('--pseudoreward-mag', default=5)
parser.add_argument('--pseudoreward-std', default=5)

parser.add_argument('--fname', type=str, default='')
parser.add_argument('--save-periodically', action='store_true')
parser.add_argument('--backend', type=str, default='tensorflow', choices=['tensorflow', 'pytorch'])
parser.add_argument('--max-steps', type=int, default=100, help='maximum steps per episode')

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


env = create_env(goal_distance=args.train_goal_distance, args=args, max_steps=args.max_steps)
eval_env = create_env(goal_distance=args.train_goal_distance, args=args, eval_mode=True, max_steps=args.max_steps)

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


if not os.path.exists('models_{}'.format(args.backend)):
    os.makedirs('models_{}'.format(args.backend))

# If no model name specified, generate based on parameters
if args.fname == '':
    fname = 'models_{}/{}_{}_{}dim_{}x{}_{}sensors_{}seed_{}steps_{}gd_{}ms'.format(
        args.backend,
        args.env_size, args.algo, args.ssp_dim,
        args.hidden_size, args.hidden_layers,
        args.n_sensors, args.seed, args.n_steps, args.train_goal_distance,
        args.max_steps
    )
    if args.curriculum:
        fname += '_cur'
    if args.use_open_env:
        fname += '_open'
    if args.regular_coordinates:
        fname += '_2d'
    if args.discrete_actions > 0:
        fname += '_disc{}'.format(args.discrete_actions)
    if args.continuous == 0:
        fname += '_discobs'
else:
    fname = args.fname

# load or train model
if os.path.exists(fname + '.zip'):
    print("Loading {} Model".format(args.algo))
    model = Algo.load(fname)
else:
    print("Training {} Model".format(args.algo))
    # policy_kwargs = dict(layers=[args.ssp_dim])
    policy_kwargs = dict(net_arch=[args.hidden_size]*args.hidden_layers)
    model = Algo('MlpPolicy', env, verbose=args.verbose, policy_kwargs=policy_kwargs)

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
            cur_envs.append(create_env(goal_distance=i, args=args, max_steps=args.max_steps))
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

        if args.save_periodically:
            np.savez(fname + '.npz', eval_data=eval_data)
            model.save(fname + '_steps_{}'.format(total_steps))

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
                model.set_env(cur_envs[min(goal_distance-1, max_dist-2)])

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
    env = create_env(goal_distance=args.demo_goal_distance, args=args, max_steps=100)
    # demo model
    obs = env.reset()
    for i in range(args.n_demo_steps):
        action, _states = model.predict(obs, deterministic=args.deterministic_demo)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        time.sleep(0.001)

env.close()
