from env_interface import EnvInterface
import numpy as np
import argparse

import torch
from deep_rl import random_seed, ImageNormalizer, SignNormalizer, PPOAgent, Config, CategoricalActorCriticNet, \
    NatureConvBody, get_logger
from deep_rl.utils import run_steps, mkdir, get_default_log_dir #generate_tag
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
from deep_rl.component.envs import DummyVecEnv
import os
from baselines import bench


def main():
    parser = argparse.ArgumentParser("Train an agent on a static deepmind_lab environment")
    parser.add_argument('--episode-length', type=int, default=1000,
                        help='Number of steps in an episode')

    parser.add_argument('--level-script', type=str,
                        default='contributed/dmlab30/explore_goal_locations_small',
                        help='The environment level script to load')
    parser.add_argument('--observation-type', type=str, default='vision', choices=['vision', 'pose', 'both'])

    args = parser.parse_args()

    use_vision = False
    use_pos = False
    if args.observation_type == 'vision':
        use_vision = True
    elif args.observation_type == 'pose':
        use_pos = True
    elif args.observation_type == 'both':
        use_vision = True
        use_pos = True

    # env = EnvInterface(
    #     use_vision=use_vision,
    #     use_pos=use_pos,
    #     episode_length=args.episode_length,
    #     level=args.level_script,
    # )


    # TODO: need a hybrid of a conv net and a fully connected net to combine vision with the x,y,th info if using both
    #       otherwise just a conv net if using pure vision

    if args.observation_type == 'vision':
        ppo_pixel(
            log_name='ppo-dmlab-image',
            render=False,
        )


def ppo_pixel(log_name='ppo-dmlab-image', render=False):

    config = Config()
    log_dir = get_default_log_dir(ppo_pixel.__name__)

    config.num_workers = 8

    config.task_fn = lambda: Task(
        use_vision=True,
        use_pos=False,
        num_envs=config.num_workers,
        render=render,
    )
    config.eval_env = Task(
        use_vision=True,
        use_pos=False,
        num_envs=config.num_workers,
        log_dir=log_dir,
        render=render,
    )

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, NatureConvBody(in_channels=3)
    )
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.logger = get_logger(tag=log_name)
    config.tag = log_name  # this name must be unique. Anything with the same name will be overwritten
    config.max_steps = int(2e7)
    run_steps(PPOAgent(config))


# overwite of make_env from DeepRL
# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(use_vision, use_pos, seed, rank, log_dir, episode_length,
             level_script='contributed/dmlab30/explore_goal_locations_small'):
    def _thunk():
        env = EnvInterface(
            use_vision=use_vision,
            use_pos=use_pos,
            episode_length=episode_length,
            level=level_script,
        )
        random_seed(seed)

        env.seed(seed + rank)

        if log_dir is not None:
            # env = Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)
            env = bench.Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)

        return env

    return _thunk


# Overwrite of the Task class from DeepRL
class Task:
    def __init__(self,
                 use_vision,
                 use_pos,
                 episode_length=1000,
                 level_script='contributed/dmlab30/explore_goal_locations_small',
                 render=False,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 seed=np.random.randint(int(1e5))):
        self.render = render
        if log_dir is not None:
            mkdir(log_dir)
        # TODO FIXME: need log and seed stuff as input here
        envs = [make_env(
            use_vision=use_vision, use_pos=use_pos, episode_length=episode_length, level_script=level_script,
            seed=seed, rank=i, log_dir=log_dir,
        ) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = 'ssp_env'
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        if self.render:
            # self.env.render()
            # Need to access a specific env in the DummyVecEnv. Only one is used anyway
            self.env.envs[0].render()
        return self.env.step(actions)


if __name__ == "__main__":
    main()
