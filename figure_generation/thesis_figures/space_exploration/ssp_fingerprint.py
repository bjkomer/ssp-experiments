import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spatial_semantic_pointers.utils import get_heatmap_vectors, get_fixed_dim_sub_toriod_axes, make_good_unitary, \
    encode_point, ssp_to_loc_v, make_fixed_dim_periodic_axis, get_axes
from ssp_navigation.utils.models import LearnedSSPEncoding
import torch
import nengo_spa as spa


def phi_mag_and_dir(X, Y):
    n_phi = ((len(X) - 1) // 2)
    xf = np.fft.fft(X)
    yf = np.fft.fft(Y)

    md = np.zeros((n_phi, 2))
    phi_pos = np.zeros((n_phi, 2))

    for i in range(n_phi):
        phi_pos[i, 0] = np.log(xf[i + 1]).imag
        phi_pos[i, 1] = np.log(yf[i + 1]).imag
        md[i, 0] = np.sqrt(phi_pos[i, 0]**2 + phi_pos[i, 1]**2)
        md[i, 1] = np.arctan2(phi_pos[i, 1], phi_pos[i, 0])

    return md, phi_pos


def magnitude_histogram(md, ax):
    # sns.distplot(md[:, 0], ax=ax, rug=True, hist=False)
    sns.distplot(md[:, 0], ax=ax, rug=True, hist=True)
    # sns.distplot(md[:, 0], ax=ax, rug=True, hist=True, bins=30)
    # sns.distplot(md[:, 0], ax=ax, rug=True, hist=True, bins=20)
    ax.set_xlim([0, np.sqrt(2)*np.pi])
    # ax.set_title('Magnitude Histogram')
    # ax.set_xticks([0, np.pi])


def direction_histogram(md, ax):
    # sns.distplot(md[:, 1], ax=ax, rug=True, hist=False)
    sns.distplot(md[:, 1], ax=ax, rug=True, hist=True)
    ax.set_title('Direction Histogram')


def mag_dir_histogram(md, ax, res=128):
    xs = np.linspace(-np.pi, np.pi, res)
    ys = np.linspace(-np.pi, np.pi, res)
    H, xedges, yedges = np.histogram2d(md[:, 0], md[:, 1], bins=(xs, ys))
    ax.imshow(H)


def gaussian_2d(x, y, sigma, meshgrid):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)


def gauss_image(phi_pos, ax, sigma=.1, res=256):
    xs = np.linspace(-np.pi, np.pi, res)
    ys = np.linspace(-np.pi, np.pi, res)
    mg = np.meshgrid(xs, ys)
    img = np.zeros((res, res))

    for i in range(phi_pos.shape[0]):
        img += gaussian_2d(x=phi_pos[i, 0], y=phi_pos[i, 1], sigma=sigma, meshgrid=mg)

    im = ax.imshow(img)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_heatmap(X, Y, xs, ys, ax):
    sim = np.zeros((len(xs), len(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            sim[i, j] = encode_point(x, y, spa.SemanticPointer(data=X), spa.SemanticPointer(data=Y)).v[0]

    im = ax.imshow(sim, vmin=0, vmax=1)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def get_axes_from_network(fname, dim, hidden_size=1024):
    model = LearnedSSPEncoding(
        input_size=2, encoding_size=dim, maze_id_size=256,
        hidden_size=hidden_size, output_size=2, n_layers=1, dropout_fraction=0.0
    )

    model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage), strict=False)

    phis = model.encoding_layer.phis.detach().numpy()

    xf = np.ones((dim,), dtype='complex64')
    yf = np.ones((dim,), dtype='complex64')

    n_phi = ((dim - 1) // 2)

    xf[1:n_phi+1] = np.exp(1.j*phis[0, :])
    yf[1:n_phi + 1] = np.exp(1.j*phis[1, :])

    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])

    X = np.fft.ifft(xf).real
    Y = np.fft.ifft(yf).real

    assert np.allclose(np.linalg.norm(X), 1)
    assert np.allclose(np.linalg.norm(Y), 1)

    return X, Y


def final():
    fig, ax = plt.subplots(5, 4, figsize=(8, 6), tight_layout=True)
    fig2, ax2 = plt.subplots(5, 3, figsize=(5, 4), tight_layout=True)

    dim = 256
    res = 32
    sigma = .15
    n_toroid = (dim - 1) // 2

    for type_index in range(5):
        phi_mag_total = np.zeros((n_toroid*3, 1))
        for seed in range(3):
            if type_index == 0:
                # random SSP
                rng = np.random.RandomState(seed=seed)
                X = make_good_unitary(dim, rng=rng)
                Y = make_good_unitary(dim, rng=rng)

                md, phi_pos = phi_mag_and_dir(X.v, Y.v)
            elif type_index == 1:
                # sub-toroid SSP
                rng = np.random.RandomState(seed=seed)
                X, Y = get_fixed_dim_sub_toriod_axes(
                    dim=dim,
                    n_proj=3,
                    scale_ratio=0,
                    scale_start_index=0,
                    rng=rng,
                    eps=0.001,
                )

                md, phi_pos = phi_mag_and_dir(X.v, Y.v)
            elif type_index == 2:
                # learned SSP no regularization
                X, Y = get_axes_from_network(
                    fname='learned_ssp_models/no_reg_model_1layer_1024hs_seed{}.pt'.format(seed),
                    # fname='learned_ssp_models/reg_proper_model_1layer_1024hs_seed{}.pt'.format(seed),
                    # fname='learned_ssp_models/reg_proper_noise_model_1layer_1024hs_seed{}.pt'.format(seed),
                    # fname='learned_ssp_models/reg_proper_phi_decay_model_1layer_1024hs_seed{}.pt'.format(seed),
                    dim=dim
                )
                md, phi_pos = phi_mag_and_dir(X, Y)
            elif type_index == 3:
                # learned SSP just weight regularization
                X, Y = get_axes_from_network(
                    fname='learned_ssp_models/reg_proper_model_1layer_1024hs_seed{}.pt'.format(seed),
                    dim=dim
                )
                md, phi_pos = phi_mag_and_dir(X, Y)
            elif type_index == 4:
                # learned SSP weight and phi regularization
                X, Y = get_axes_from_network(
                    fname='learned_ssp_models/reg_proper_phi_decay_model_1layer_1024hs_seed{}.pt'.format(seed),
                    dim=dim
                )
                md, phi_pos = phi_mag_and_dir(X, Y)

            gauss_image(phi_pos, ax[type_index, seed], sigma=sigma, res=res)
            if type_index < 2:
                X = X.v
                Y = Y.v
            im = plot_heatmap(X, Y, np.linspace(-5, 5, 128), np.linspace(-5, 5, 128), ax2[type_index, seed])
            phi_mag_total[seed * n_toroid:(seed + 1) * n_toroid, 0] = md[:, 0]

            if seed == 0:
                if type_index == 0:
                    # label = 'Fixed SSP'
                    label = 'A     '
                elif type_index == 1:
                    # label = 'Fixed Grid SSP'
                    label = 'B     '
                elif type_index == 2:
                    # label = 'Learned SSP'
                    label = 'C     '
                elif type_index == 3:
                    # label = 'Learned SSP'
                    label = 'D     '
                elif type_index == 4:
                    # label = 'Learned SSP'
                    label = 'E     '
                # ax[type_index, seed].set_ylabel(
                #     label, rotation=90, fontsize=18,
                #     # position=(0, .4)
                # )
                ax[type_index, seed].set_ylabel(
                    label, rotation=0, fontsize=18,
                    position=(0, .4)
                )
                ax2[type_index, seed].set_ylabel(
                    label, rotation=0, fontsize=18,
                    position=(0, .4)
                )

        # histogram at the bottom
        magnitude_histogram(phi_mag_total, ax[type_index, 3])

        # if type_index == 0:
        #
        #     ax[type_index, 1].set_title('Phi Magnitude and Direction')
        #
        #     ax[type_index, 3].set_title('Phi Magnitude Histogram')

    # cbar_ax = fig2.add_axes([0.85, 0.05, 0.05, 0.85])
    # add_axes([left, bottom, width, height])
    cbar_ax = fig2.add_axes([0.85, 0.05, 0.05, 0.90])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def debug():
    seed = 14
    dim = 256
    # dim = 1024
    res = 256 #32
    res = 32

    rng = np.random.RandomState(seed=seed)

    X, Y = get_fixed_dim_sub_toriod_axes(
        dim=dim,
        n_proj=3,
        scale_ratio=0,
        scale_start_index=0,
        rng=rng,
        eps=0.001,
    )

    X, Y = get_axes_from_network(fname='learned_ssp_models/model_1layer_1024hs_seed1.pt', dim=dim)

    # X, Y = get_axes(dim=dim, n=3, seed=13, period=0, optimal_phi=False)

    X = make_good_unitary(dim, rng=rng)
    Y = make_good_unitary(dim, rng=rng)

    # X = make_fixed_dim_periodic_axis(
    #     dim=dim, period=6, phase=0, frequency=1, eps=1e-3, rng=rng, flip=False, random_phases=False,
    # )
    # Y = make_fixed_dim_periodic_axis(
    #     dim=dim, period=6, phase=0, frequency=1, eps=1e-3, rng=rng, flip=False, random_phases=False,
    # )

    md, phi_pos = phi_mag_and_dir(X.v, Y.v)
    # md, phi_pos = phi_mag_and_dir(X, Y)

    fig, ax = plt.subplots(1, 5, figsize=(10, 3))

    magnitude_histogram(md, ax[0])
    direction_histogram(md, ax[1])

    mag_dir_histogram(md, ax[2], res=res)
    mag_dir_histogram(phi_pos, ax[3], res=res)
    gauss_image(phi_pos, ax[4], sigma=.15)

    print("Mean Location: ({},{})".format(
        np.round(np.mean(phi_pos[:, 0]), 2), np.round(np.mean(phi_pos[:, 1]), 2)
    ))

    plt.show()

if __name__ == '__main__':
    final()
