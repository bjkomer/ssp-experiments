import numpy as np
import argparse
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.maze_generation import generate_maze
from agents import RandomTrajectoryAgent


parser = argparse.ArgumentParser('View a random agent on an environment')

# parser = add_map_arguments(parser)

parser.add_argument('--goal-distance', type=int, default=0, help='distance of the goal from the start location')
parser.add_argument('--map-style', type=str, default='blocks', choices=['blocks', 'maze'], help='type of map layout')

args = parser.parse_args()
# params = args_to_dict(args)

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': 36,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': 13,
    'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': 200, #1000,
    'max_lin_vel': 5,
    'max_ang_vel': 5,
    'dt': 0.1,
    'full_map_obs': False,
    'pob': 0,
    'n_grid_cells': 0,
    'heading': 'none',
    'location': 'none',
    'goal_loc': 'none',
    'goal_vec': 'none',
    'bc_n_ring': 0,
    'hd_n_cells': 0,
    'csp_dim': 0,
    'goal_csp': False,
    'agent_csp': False,

    'goal_distance': args.goal_distance  # 0 means completely random
}

obs_dict = generate_obs_dict(params)

np.random.seed(params['seed'])

map_array = generate_maze(map_style=params['map_style'], side_len=params['map_size'])

# map_array = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])

env = GridWorldEnv(
    map_array=map_array,
    # object_locations=object_locations,
    observations=obs_dict,
    movement_type=params['movement_type'],
    max_lin_vel=params['max_lin_vel'],
    max_ang_vel=params['max_ang_vel'],
    continuous=params['continuous'],
    max_steps=params['episode_length'],
    fixed_episode_length=params['fixed_episode_length'],
    dt=params['dt'],
    screen_width=300,
    screen_height=300,
)

agent = RandomTrajectoryAgent()


num_episodes = 10
time_steps = 10000#100
returns = np.zeros((num_episodes,))
for e in range(num_episodes):
    obs = env.reset(goal_distance=params['goal_distance'])
    for s in range(params['episode_length']):
        env.render()
        env._render_extras()

        action = agent.act(obs)

        obs, reward, done, info = env.step(action)
        # print(obs)
        returns[e] += reward
        # if reward != 0:
        #    print(reward)
        # time.sleep(dt)
        # ignoring done flag and not using fixed episodes, which effectively means there is no goal
        # if done:
        #     break

print(returns)
