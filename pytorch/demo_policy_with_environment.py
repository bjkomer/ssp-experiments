import numpy as np
import argparse
import torch
from models import FeedForward
from env_utils import WrappedSSPEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser('View a policy running on an enviromnent')

    parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_10mazes_25goals_64res_13seed.npz')
    parser.add_argument('--model', type=str, default='', help='Saved model to load from')
    parser.add_argument('--map-index', type=int, default=0, help='Index for picking which map in the dataset to use')

    args = parser.parse_args()

    # Load a dataset to get the mazes/goals and axes the policy was trained on
    data = np.load(args.dataset)

    dim = data['x_axis_sp'].shape[0]

    env = WrappedSSPEnv(data=data, map_index=args.map_index)

    # input is maze, loc, goal ssps, output is 2D direction to move
    model = FeedForward(input_size=dim * 3, output_size=2)

    if args.model:
        model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()

    num_episodes = 10
    time_steps = 1000
    returns = np.zeros((num_episodes,))
    with torch.no_grad():

        for e in range(num_episodes):
            obs = env.reset()
            for s in range(time_steps):

                action = model(torch.Tensor(obs)).squeeze(0).numpy()

                # Add some noise to the action
                action += np.random.normal(size=2)*.75

                obs, reward, done, info = env.step(action)

                returns[e] += reward

                env.render()

                if done:
                    break

    print(returns)
