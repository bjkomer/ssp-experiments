import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, \
    get_heatmap_vectors, encode_point_hex
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d
import argparse
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir, orthogonal_hex_dir_7dim

parser = argparse.ArgumentParser("Gather data on spatial activations of neurons with different encoders")

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5.)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--encoder-type', type=str, default='grid_cell',
                    choices=['place_cell', 'grid_cell', 'band', 'default', 'mixed'])
parser.add_argument('--duration', type=int, default=70)
parser.add_argument('--neurons-per-dim', type=int, default=15)
parser.add_argument('--intercepts', type=float, default=0.2)
parser.add_argument('--max-rates', type=float, default=50)
parser.add_argument('--trajectory-type', type=str, default='hilbert', choices=['random', 'hilbert', 'many-random'])

args = parser.parse_args()


rng = np.random.RandomState(seed=args.seed)

# phis = (np.pi / 2., np.pi/3., np.pi/5., np.pi/7.)
# phis = (np.pi / 2., np.pi/3., np.pi/5.)
phis = (np.pi*.75, np.pi / 2., np.pi/3., np.pi/5., np.pi*.4, np.pi*.6, np.pi*.15)
# angles = rng.uniform(0, 2*np.pi, size=len(phis))#(0, np.pi/3., np.pi/5.)
angles = (0, np.pi*.3, np.pi*.2, np.pi*.4, np.pi*.1, np.pi*.5, np.pi*.7)
X, Y = orthogonal_hex_dir(phis=phis, angles=angles)
dim = len(X.v)

noise_process = nengo.processes.WhiteNoise(
    dist=nengo.dists.Gaussian(0, 0.0001), seed=1)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


def trajectory_generation(
        trajectory_steps, lin_vel_rayleigh_scale=0.13, rot_vel_std=330,
        limit_high=5., limit_low=-5., perimeter_distance=0.03, perimeter_vel_reduction=.25, dt=0.02
):

    turn_vel = 2 * rot_vel_std * np.pi / 180

    positions = np.zeros((trajectory_steps, 2))
    angles = np.zeros((trajectory_steps,))
    lin_vels = np.zeros((trajectory_steps,))
    ang_vels = np.zeros((trajectory_steps,))

    # velocity in x and y
    cartesian_vels = np.zeros((trajectory_steps, 2))

    positions[0, :] = np.random.uniform(low=limit_low, high=limit_high, size=2)
    angles[0] = np.random.uniform(low=-np.pi, high=np.pi)
    # precompute linear and angular velocities used
    lin_vels[:] = np.random.rayleigh(scale=lin_vel_rayleigh_scale, size=trajectory_steps)
    ang_vels[:] = np.random.normal(loc=0, scale=rot_vel_std, size=trajectory_steps) * np.pi / 180

    for s in range(1, trajectory_steps):

        # TODO: make sure this can handle the case of moving into a corner
        # TODO: doublecheck this is correct
        dist_wall_x = min(limit_low - positions[s - 1, 0], limit_high - positions[s - 1, 0])
        dist_wall_y = min(limit_low - positions[s - 1, 1], limit_high - positions[s - 1, 1])
        # Find which of the four walls is closest, and calculate the angle based on that wall

        if dist_wall_x == limit_low - positions[s - 1, 0]:
            angle_wall_x = angles[s - 1] - np.pi
        elif dist_wall_x == limit_high - positions[s - 1, 0]:
            angle_wall_x = angles[s - 1]

        if dist_wall_y == limit_low - positions[s - 1, 1]:
            angle_wall_y = angles[s - 1] + np.pi / 2
        elif dist_wall_y == limit_high - positions[s - 1, 1]:
            angle_wall_y = angles[s - 1] - np.pi / 2

        if angle_wall_x > np.pi:
            angle_wall_x -= 2 * np.pi
        elif angle_wall_x < -np.pi:
            angle_wall_x += 2 * np.pi

        if angle_wall_y > np.pi:
            angle_wall_y -= 2 * np.pi
        elif angle_wall_y < -np.pi:
            angle_wall_y += 2 * np.pi

        if abs(dist_wall_x) < perimeter_distance and abs(angle_wall_x) < np.pi / 2:
            # modify angular velocity to turn away from the wall
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall_x / abs(angle_wall_x) * (np.pi / 2) / args.dt
            ang_vels[s] = ang_vels[s] + angle_wall_x / abs(angle_wall_x) * turn_vel
            # slow down linear velocity
            lin_vels[s] = lin_vels[s] * perimeter_vel_reduction
            # lin_vels[n, s] = 0

        if abs(dist_wall_y) < perimeter_distance and abs(angle_wall_y) < np.pi / 2:
            # modify angular velocity to turn away from the wall
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall_y / abs(angle_wall_y) * (np.pi / 2) / args.dt
            ang_vels[s] = ang_vels[s] + angle_wall_y / abs(angle_wall_y) * turn_vel
            # slow down linear velocity
            lin_vels[s] = lin_vels[s] * perimeter_vel_reduction
            # lin_vels[n, s] = 0

        cartesian_vels[s, 0] = np.cos(angles[s - 1]) * lin_vels[s]
        cartesian_vels[s, 1] = np.sin(angles[s - 1]) * lin_vels[s]
        positions[s, 0] = positions[s - 1, 0] + cartesian_vels[s, 0] * dt
        positions[s, 1] = positions[s - 1, 1] + cartesian_vels[s, 1] * dt
        angles[s] = angles[s - 1] + ang_vels[s] * dt
        if angles[s] > np.pi:
            angles[s] -= 2 * np.pi
        elif angles[s] < -np.pi:
            angles[s] += 2 * np.pi

    return positions


model = nengo.Network(seed=args.seed)


dt = 0.001
n_samples = int(args.duration / dt)

# set of positions to visit on a space filling curve, one for each timestep
if args.trajectory_type == 'hilbert':
    # positions = hilbert_2d(-args.limit, args.limit, n_samples, rng, p=8, N=2, normal_std=0)
    positions = hilbert_2d(-args.limit, args.limit, n_samples, rng, p=6, N=2, normal_std=0)
elif args.trajectory_type == 'random':
    positions = trajectory_generation(n_samples, limit_low=-args.limit, limit_high=args.limit)
elif args.trajectory_type == 'many-random':
    positions = np.zeros((n_samples, 2))
    step = n_samples // 100
    for i in range(100):
        positions[i*step:(i+1)*step] = trajectory_generation(step, limit_low=-args.limit, limit_high=args.limit)
else:
    raise NotImplementedError

n_neurons = dim * args.neurons_per_dim

rng = np.random.RandomState(seed=args.seed)

# preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)
preferred_locations = rng.uniform(-args.limit, args.limit, size=(n_neurons, 2))

# also record which class each encoder is in
# (toroid_index, band_index, mix_index)
metadata = np.zeros((n_neurons, 3))

encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
encoders_mixed = np.zeros((n_neurons, dim))
mixed_intercepts = []
for n in range(n_neurons):
    ind = rng.randint(0, len(phis))
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])

    encoders_grid_cell[n, :] = grid_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind
    )

    band_ind = rng.randint(0, 3)
    encoders_band_cell[n, :] = band_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind,
        band_index=band_ind
    )

    mix_ind = rng.randint(0, 3)
    if mix_ind == 0:
        encoders_mixed[n, :] = encoders_place_cell[n, :]
        mixed_intercepts.append(.4)
    elif mix_ind == 1:
        encoders_mixed[n, :] = encoders_grid_cell[n, :]
        mixed_intercepts.append(.2)
    elif mix_ind == 2:
        encoders_mixed[n, :] = encoders_band_cell[n, :]
        mixed_intercepts.append(0.)

    metadata[n, 0] = ind
    metadata[n, 1] = band_ind
    metadata[n, 2] = mix_ind

# model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    pos_2d = nengo.Node(lambda t: positions[min(int(np.floor(t/dt)), n_samples - 1)], size_in=0, size_out=2)
    ssp_loc = nengo.Ensemble(
        n_neurons=n_neurons, dimensions=dim,
        noise=noise_process,
        max_rates=nengo.dists.Uniform(args.max_rates, args.max_rates)
    )

    if args.encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif args.encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
        ssp_loc.eval_points = encoders_place_cell
    elif args.encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
        ssp_loc.eval_points = encoders_band_cell
    elif args.encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell
        ssp_loc.eval_points = encoders_grid_cell
    elif args.encoder_type == 'mixed':
        ssp_loc.encoders = encoders_mixed
        ssp_loc.eval_points = encoders_mixed
    else:
        raise NotImplementedError
    # TODO: add mixed encoder option

    ssp_loc.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])

    ssp_loc.intercepts = mixed_intercepts #[args.intercepts]*n_neurons
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(pos_2d, ssp_loc, function=to_ssp)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
        pos_2d_p = nengo.Probe(pos_2d)
    else:
        res = 128
        xs = np.linspace(-args.limit, args.limit, res)
        ys = np.linspace(-args.limit, args.limit, res)
        heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
        # create a visualization of the heatmap
        heatmap_node = nengo.Node(
            SpatialHeatmap(
                heatmap_vectors,
                xs,
                ys,
                cmap='plasma',
                vmin=-1,  # None,
                vmax=1,  # None,
            ),
            size_in=dim,
            size_out=0,
        )

        nengo.Connection(ssp_loc, heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(args.duration)

    folder = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output'
    fname = '{}/output_orth_hex_toroid_d{}_{}_{}hz_{}_traj_{}limit_{}s.npz'.format(
        folder, dim, args.encoder_type, args.max_rates, args.trajectory_type, args.limit, args.duration
    )
    print("saving to: {}".format(fname))
    np.savez(
        fname,
        spikes=sim.data[spikes_p],
        pos_2d=sim.data[pos_2d_p],
        metadata=metadata,
    )
