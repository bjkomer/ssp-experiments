import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import Environment, spatial_heatmap
from ssp_navigation.utils.encodings import get_encoding_function
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=20)
# parser.add_argument('--dataset', type=str,
#                     default='../path_integration/data/path_integration_raw_trajectories_100t_15s_seed13.npz')
parser.add_argument('--spatial-encoding', type=str, default='pc-gauss',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding'
                    ])
parser.add_argument('--res', type=int, default=64, help='resolution of the spatial heatmap')

parser.add_argument('--limit-low', type=float, default=0.0)
parser.add_argument('--limit-high', type=float, default=2.2)

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

encoding_func, dim = get_encoding_function(args, limit_low=args.limit_low, limit_high=args.limit_high)

env = Environment(encoding_func=encoding_func, limit_low=args.limit_low, limit_high=args.limit_high)

xs = np.linspace(0, 2.2, args.res)
ys = np.linspace(0, 2.2, args.res)

show_pca = True

if True:  # live updates
    n_steps = 1000

    # fig, ax = plt.subplots()

    pca = PCA(n_components=args.n_components)

    all_activations = None
    all_pos = None

    fig_traj = plt.figure()
    ax = plt.axes(xlim=(args.limit_low, args.limit_high), ylim=(args.limit_low, args.limit_high))
    line, = ax.plot([], [], lw=3)
    x = []
    y = []

    fig_pca = plt.figure()
    # ax_pca = plt.axes(xlim=(0, args.n_components), ylim=(0, args.n_components))
    ax_pca = plt.axes()
    image = ax_pca.imshow(np.random.uniform(0, 1, size=(args.n_components, args.n_components)))

    plt.show(block=False)

    for n in range(n_steps):
        activations, pos = env.step()

        if all_activations is None:
            all_activations = activations.reshape((1, dim)).copy()
        else:
            all_activations = np.append(all_activations, activations.reshape((1, dim)), axis=0)

        if all_pos is None:
            all_pos = pos.reshape((1, 2)).copy()
        else:
            all_pos = np.append(all_pos, pos.reshape((1, 2)), axis=0)

        if show_pca:
            # pca.fit(all_activations)
            # transformed_activations = pca.transform(all_activations)

            heatmap = spatial_heatmap(all_activations, all_pos, xs, ys)
            # heatmap = spatial_heatmap(transformed_activations, all_pos, xs, ys)
            # print(heatmap.shape)
            print(np.max(heatmap[:, :, :]))

            combined_heatmap = np.mean(heatmap, axis=0)
            print(combined_heatmap)
            image.set_data(combined_heatmap)
            # image.set_data(heatmap[0, :, :])
            fig_pca.canvas.draw()
            # if (n > 0) and (n % 20 == 0):
            #     pca.fit(all_activations)

        # x.append(pos[0])
        # y.append(pos[1])
        line.set_data(all_pos[:, 0], all_pos[:, 1])
        fig_traj.canvas.draw()

if False:  # create animation
    fig = plt.figure()
    ax = plt.axes(xlim=(args.limit_low, args.limit_high), ylim=(args.limit_low, args.limit_high))
    line, = ax.plot([], [], lw=3)
    x = []
    y = []

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        activations, pos = env.step()
        x.append(pos[0])
        y.append(pos[1])
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)

    anim.save('trajectory.gif', writer='imagemagick')


