import numpy as np
import argparse
import sys
from env_utils import WrappedSSPEnv
from gridworlds.getch import getch

parser = argparse.ArgumentParser('Control the agent with a keyboard')

parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_10mazes_25goals_64res_13seed.npz')
parser.add_argument('--map-index', type=int, default=0, help='Index for picking which map in the dataset to use')
parser.add_argument('--noise', type=float, default=0.75, help='Magnitude of gaussian noise to add to the actions')
parser.add_argument('--random-goals', action='store_true',
                    help='use random goal locations rather than those in the datasets')

args = parser.parse_args()

# Load a dataset to get the mazes/goals and axes the policy was trained on
data = np.load(args.dataset)

dim = data['x_axis_sp'].shape[0]
n_mazes = data['fine_mazes'].shape[0]

# using WASD instead of arrow keys for consistency
UP = 119  # W
LEFT = 97  # A
DOWN = 115  # S
RIGHT = 100  # D

SHUTDOWN = 99  # C


def keyboard_action():
    action_str = getch()

    if ord(action_str) == UP:
        action = np.array([.75, 0])
    elif ord(action_str) == DOWN:
        action = np.array([-.75, 0])
    elif ord(action_str) == LEFT:
        action = np.array([0, -.25])
    elif ord(action_str) == RIGHT:
        action = np.array([0, .25])
    else:
        action = np.array([0, 0])

    # Set a signal to close the environment
    if ord(action_str) == SHUTDOWN:
        action = None

    return action


env = WrappedSSPEnv(
    data=data,
    map_index=args.map_index,
    map_encoding='ssp',
    random_object_locations=args.random_goals
)

num_episodes = 10
time_steps = 1000
returns = np.zeros((num_episodes,))

for e in range(num_episodes):
    obs = env.reset()
    for s in range(time_steps):

        action = keyboard_action()

        # If a specific key is pressed, close the script
        if action is None:
            env.close()
            sys.exit(0)

        # Add some noise to the action
        # action += np.random.normal(size=2) * args.noise

        obs, reward, done, info = env.step(action)

        returns[e] += reward

        env.render()

        if done:
            break

print(returns)
