import numpy as np
import argparse
import torch
from models import FeedForward
from env_utils import WrappedSSPEnv

# DeepRL uses * imports everywhere. Using them here so the code matches even though they are awful
from deep_rl import *
from deep_rl.utils import *
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
from deep_rl.component.envs import DummyVecEnv
import os
from baselines import bench
import gym

# overwite of make_env from DeepRL
# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(data, map_index, seed, rank, log_dir, episode_life=True):
    def _thunk():
        env = WrappedSSPEnv(data=data, map_index=map_index)
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
                 data,
                 map_index,
                 render=False,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        self.render = render
        if log_dir is not None:
            mkdir(log_dir)
        # TODO FIXME: need log and seed stuff as input here
        envs = [make_env(
            data=data, map_index=map_index, seed=seed, rank=i, log_dir=log_dir, episode_life=episode_life
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


def modified_gaussian_actor_critic_net(state_dim, action_dim, model_params, critic_body, load_given_params=True):

    actor_body = FCBody(state_dim, hidden_units=(512,), gate=F.relu)
    if load_given_params:
        # overwrite the weights of the actor body
        actor_body.layers[0].weight = torch.nn.Parameter(model_params['input_layer.weight'])
        actor_body.layers[0].bias = torch.nn.Parameter(model_params['input_layer.bias'])

    net = GaussianActorCriticNet(
        state_dim, action_dim,
        actor_body=actor_body,
        critic_body=critic_body
    )

    if load_given_params:
        # overwrite the weights of the action output
        net.network.fc_action.weight = torch.nn.Parameter(model_params['output_layer.weight'])
        net.network.fc_action.bias = torch.nn.Parameter(model_params['output_layer.bias'])

    return net


def ppo_continuous(data, map_index, model_params, render=False):
    config = Config()
    log_dir = get_default_log_dir(ppo_continuous.__name__)
    # config.task_fn = lambda: Task(name)
    # config.eval_env = Task(name, log_dir=log_dir)
    # config.task_fn = lambda: WrappedSSPEnv(data=data, map_index=map_index)
    # config.eval_env = WrappedSSPEnv(data=data, map_index=map_index)
    config.task_fn = lambda: Task(data=data, map_index=map_index, render=render)
    config.eval_env = Task(data=data, map_index=map_index, log_dir=log_dir, render=render)

    # config.network_fn = lambda: GaussianActorCriticNet(
    #     config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=F.tanh),
    #     critic_body=FCBody(config.state_dim, gate=F.tanh))
    # config.network_fn = lambda: GaussianActorCriticNet(
    #     config.state_dim, config.action_dim, actor_body=model,
    #     critic_body=FCBody(config.state_dim, gate=F.tanh))
    config.network_fn = lambda: modified_gaussian_actor_critic_net(
        config.state_dim, config.action_dim, model_params=model_params,
        critic_body=FCBody(config.state_dim, gate=F.tanh),
        # load_given_params=True,
        load_given_params=False,
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 1e5
    config.state_normalizer = MeanStdNormalizer()
    config.logger = get_logger()
    run_steps(PPOAgent(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('View a policy running on an enviromnent')

    parser.add_argument('--dataset', type=str,
                        default='maze_datasets/maze_dataset_maze_style_50mazes_25goals_64res_13size_13seed.npz')
    parser.add_argument('--model', type=str,
                        default='multi_maze_solve_function/v2/ssp/maze/mid_lr_mom_much_more_data/Mar07_15-23-16/model_epoch_100.pt',
                        help='Saved model to load from')
    parser.add_argument('--map-index', type=int, default=0, help='Index for picking which map in the dataset to use')

    args = parser.parse_args()

    # Load a dataset to get the mazes/goals and axes the policy was trained on
    data = np.load(args.dataset)

    dim = data['x_axis_sp'].shape[0]

    # env = WrappedSSPEnv(data=data, map_index=args.map_index)

    # # input is maze, loc, goal ssps, output is 2D direction to move
    # model = FeedForward(input_size=dim * 3, output_size=2)

    # if args.model:
    #     model.load_state_dict(torch.load(args.model), strict=False)

    model_params = torch.load(args.model)

    # # print(torch.load(args.model))
    # model_data = torch.load(args.model)
    # print(model_data.keys())
    # for param in model_data:
    #     print(model_data[param].shape)
    #
    # assert False

    # ppo_continuous(data, args.map_index, model_params, render=True)
    ppo_continuous(data, args.map_index, model_params, render=False)
