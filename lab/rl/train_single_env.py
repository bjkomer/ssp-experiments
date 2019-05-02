from env_interface import EnvInterface
import numpy as np
import argparse

parser = argparse.ArgumentParser("Train an agent on a static deepmind_lab environment")
parser.add_argument('--episode-length', type=int, default=1000,
                    help='Number of steps in an episode')

parser.add_argument('--level-script', type=str,
                    default='contributed/dmlab30/explore_goal_locations_small',
                    help='The environment level script to load')


args = parser.parse_args()

env = EnvInterface(episode_length=args.episode_length, level=args.level_script)


