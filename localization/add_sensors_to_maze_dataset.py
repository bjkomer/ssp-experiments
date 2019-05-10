import numpy as np
import argparse
from gridworlds.map_utils import generate_sensor_readings

parser = argparse.ArgumentParser('Modify an existing maze solving dataset with sensor measurements')

parser.add_argument('--original-dataset', type=str, default='', help='Original dataset to work from')
parser.add_argument('--new-dataset', type=str, default='', help='New dataset to create')
# parser.add_argument('--ssp-scaling', type=float, default=5, help='amount to multiply coordinates by before converting to SSP')
parser.add_argument('--n-sensors', type=int, default=36, help='number of distance sensors around the agent')
parser.add_argument('--fov', type=float, default=360, help='field of view of distance sensors, in degrees')
parser.add_argument('--max-dist', type=float, default=10, help='maximum distance for distance sensor')

args = parser.parse_args()

# Load contents from the original dataset
original_dataset = np.load(args.original_dataset)

xs = original_dataset['xs']
ys = original_dataset['ys']
x_axis_vec = original_dataset['x_axis_sp']
y_axis_vec = original_dataset['y_axis_sp']
coarse_mazes = original_dataset['coarse_mazes']
fine_mazes = original_dataset['fine_mazes']
solved_mazes = original_dataset['solved_mazes']
maze_sps = original_dataset['maze_sps']
goal_sps = original_dataset['goal_sps']
goals = original_dataset['goals']

n_mazes = solved_mazes.shape[0]
n_goals = solved_mazes.shape[1]
res = solved_mazes.shape[2]
coarse_maze_size = coarse_mazes.shape[1]

limit_low = xs[0]
limit_high = xs[-1]

sensor_scaling = (limit_high - limit_low) / coarse_maze_size

# Scale to the coordinates of the coarse maze, for getting distance measurements
xs_scaled = ((xs - limit_low) / (limit_high - limit_low)) * coarse_maze_size
ys_scaled = ((ys - limit_low) / (limit_high - limit_low)) * coarse_maze_size

n_sensors = args.n_sensors
fov_rad = args.fov * np.pi / 180

dist_sensors = np.zeros((n_mazes, res, res, n_sensors))

# Generate sensor readings for every location in xs and ys in each maze
for mi in range(n_mazes):
    # NOTE: only need to create the env if other 'sensors' such as boundary cells are used
    # # Generate a GridWorld environment corresponding to the current maze in order to calculate sensor measurements
    # env = GridWorldEnv(
    #     map_array=coarse_mazes[mi, :, :],
    #     movement_type='holonomic',
    #     continuous=True,
    # )
    #
    # sensors = env.get_dist_sensor_readings(
    #     state=state,
    #     n_sensors=n_sensors,
    #     fov_rad=fov_rad,
    #     max_dist=args.max_dist,
    #     normalize=False
    # )

    for xi, x in enumerate(xs_scaled):
        for yi, y in enumerate(ys_scaled):

            # Only compute measurements if not in a wall
            if fine_mazes[mi, xi, yi] == 0:
                # Compute sensor measurements and scale them based on xs and ys
                dist_sensors[mi, xi, yi, :] = generate_sensor_readings(
                    map_arr=coarse_mazes[mi, :, :],
                    n_sensors=n_sensors,
                    fov_rad=fov_rad,
                    x=x,
                    y=y,
                    th=0,
                    max_sensor_dist=args.max_dist,
                    debug_value=0,
                ) * sensor_scaling


# Save the new dataset
np.savez(
    args.new_dataset,
    dist_sensors=dist_sensors,
    # heatmap_vectors=heatmaps_vectors,
    coarse_mazes=coarse_mazes,
    fine_mazes=fine_mazes,
    solved_mazes=solved_mazes,
    maze_sps=maze_sps,
    x_axis_sp=x_axis_vec,
    y_axis_sp=y_axis_vec,
    goal_sps=goal_sps,
    goals=goals,
    xs=xs,
    ys=ys,
    max_dist=args.max_dist,
    fov=args.fov,
)
