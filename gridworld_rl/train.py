# Train with curriculum learning using the DeepRL code
import argparse
from gridworlds.envs import GridWorldEnv
from env_utils import get_env
from gridworlds.getch import getch
import numpy as np
import sys

parser = argparse.ArgumentParser("Train an agent to navigate to goals in a 2D world")

parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()


env = get_env()





if False:

    # use a random agent to see if things work
    num_episodes = 10
    time_steps = 1000
    returns = np.zeros((num_episodes,))

    # using WASD instead of arrow keys for consistency
    UP = 119  # W
    LEFT = 97  # A
    DOWN = 115  # S
    RIGHT = 100  # D

    SHUTDOWN = 99  # C


    def keyboard_action():
        action_str = getch()

        if ord(action_str) == UP:
            # action = np.array([.75, 0])
            action = np.array([1.0, 0])
        elif ord(action_str) == DOWN:
            # action = np.array([-.75, 0])
            action = np.array([-1.0, 0])
        elif ord(action_str) == LEFT:
            # action = np.array([0, -.25])
            action = np.array([0, -1.0])
        elif ord(action_str) == RIGHT:
            # action = np.array([0, .25])
            action = np.array([0, 1.0])
        else:
            action = np.array([0, 0])

        # Set a signal to close the environment
        if ord(action_str) == SHUTDOWN:
            action = None

        return action


    for e in range(num_episodes):
        obs = env.reset()
        for s in range(time_steps):

            action = keyboard_action()

            # If a specific key is pressed, close the script
            if action is None:
                env.close()
                sys.exit(0)

            # Add some noise to the action
            # action += np.random.normal(size=2) * .1

            obs, reward, done, info = env.step(action)

            returns[e] += reward

            env.render()

            if done:
                break

    print(returns)
