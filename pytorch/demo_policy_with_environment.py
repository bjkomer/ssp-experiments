import numpy as np
import argparse
import torch
import os
import json
from models import load_model, FeedForward, LearnedEncoding
from env_utils import WrappedSSPEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser('View a policy running on an enviromnent')

    parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_maze_style_50mazes_25goals_64res_13size_13seed_modified.npz')
    parser.add_argument('--model-folder', type=str, default='', help='Saved model to load from')
    parser.add_argument('--map-index', type=int, default=0, help='Index for picking which map in the dataset to use')
    parser.add_argument('--noise', type=float, default=0.75, help='Magnitude of gaussian noise to add to the actions')
    parser.add_argument('--random-goals', action='store_true', help='use random goal locations rather than those in the datasets')
    parser.add_argument('--fine-tuned-weights', type=str, default='', help='optionally replace weights with fine tuned version from this file')

    args = parser.parse_args()

    model_path = os.path.join(args.model_folder, 'model.pt')
    params_path = os.path.join(args.model_folder, 'params.json')

    # Load a dataset to get the mazes/goals and axes the policy was trained on
    data = np.load(args.dataset)

    dim = data['x_axis_sp'].shape[0]
    n_mazes = data['fine_mazes'].shape[0]

    model, map_encoding = load_model(model_path, params_path, n_mazes)

    if args.fine_tuned_weights != '':
        weights = torch.load(args.fine_tuned_weights)

        # print(weights.keys())
        # print("")
        # weights2 = torch.load(model_path)
        # print(weights2.keys())

        # Overwrite the weights with the fine-tuned ones
        model.input_layer.weight = torch.nn.Parameter(weights['network.actor_body.layers.0.weight'])
        model.input_layer.bias = torch.nn.Parameter(weights['network.actor_body.layers.0.bias'])

    model.eval()

    env = WrappedSSPEnv(
        data=data,
        map_index=args.map_index,
        map_encoding=map_encoding,
        random_object_locations=args.random_goals
    )

    num_episodes = 10
    time_steps = 1000
    returns = np.zeros((num_episodes,))
    with torch.no_grad():

        for e in range(num_episodes):
            obs = env.reset()
            for s in range(time_steps):

                action = model(torch.Tensor(obs).unsqueeze(0)).squeeze(0).numpy()

                # Add some noise to the action
                action += np.random.normal(size=2) * args.noise

                obs, reward, done, info = env.step(action)

                returns[e] += reward

                env.render()

                if done:
                    break

    print(returns)
