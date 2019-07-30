import nengo.spa as spa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, \
    generate_region_vector, get_heatmap_vectors, spatial_dot
from spatial_semantic_pointers.plots import image_svg


def plot_heatmap(sp, heatmap_vectors, name='', vmin=-1, vmax=1,
                 cmap='plasma', invert=False, origin='lower', show=True):
    # vs = np.dot(vec, heatmap_vectors)
    # vec has shape (dim) and heatmap_vectors have shape (xs, ys, dim) so the result will be (xs, ys)
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    # plt.imshow(vs, origin=origin, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)

    plt.title(name)
    if show:
        plt.imshow(vs, origin=origin, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
        plt.show()
    else:
        return plt.imshow(vs, origin=origin, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)


def one_hot_axes(D=8, xi=0, yi=0):
    xv = np.zeros((D,))
    xv[xi] = 1
    yv = np.zeros((D,))
    yv[yi] = 1
    # Note that making them unitary doesn't seem to be required for one-hot vectors
    x_axis_sp = spa.SemanticPointer(data=xv)
    x_axis_sp.make_unitary()
    y_axis_sp = spa.SemanticPointer(data=yv)
    y_axis_sp.make_unitary()

    return x_axis_sp, y_axis_sp


def spatial_plot(vs, colorbar=True, vmin=-1, vmax=1, cmap='plasma'):
    vs = vs[::-1, :]
    plt.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmax=vmax, vmin=vmin, cmap=cmap)
    if colorbar:
        plt.colorbar()


def one_hot_plot(xs, ys, x=0, y=0, D=8, xi=0, yi=0, name='Origin Point', **kwargs):
    x_axis_sp, y_axis_sp = one_hot_axes(D=D, xi=xi, yi=yi)

    point = encode_point(x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

    vs = spatial_dot(point, xs=xs, ys=ys, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

    spatial_plot(vs, **kwargs)


def get_heatmap_vectors_1d(axis_vec, xs):
    vectors = np.zeros((len(xs), len(axis_vec)))
    for i, x in enumerate(xs):
        vectors[i, :] = np.fft.ifft(np.fft.fft(axis_vec) ** x).real
    return vectors


def angle(complex_num, pos_only=False):
    val = np.angle(complex_num) * 180 / np.pi

    if pos_only:
        val[val < 0] += 360
    return val


def make_periodic_axes(dim=128,
                       spacing=4,
                       phase=0,
                       eps=1e-3, rng=np.random,
                       axis_angles=[0, 120, 240],
                       correlate_axes=True,
                       alternating_trig=False
                       ):
    # will repeat at a distance of 2*spacing
    # dimensionality will be 2*spacing

    n_phi = ((dim - 1) // 2)

    # hex-axis aligned phi values
    phi_list = np.linspace(0, np.pi, spacing + 1)[1:-1]

    # choose random phis
    phi = rng.choice(phi_list, replace=True, size=n_phi)
    #     phix = rng.choice(phi_list, replace=True, size=n_phi)
    #     phiy = rng.choice(phi_list, replace=True, size=n_phi)

    # choose one of the 3 axes randomly
    axes_x = rng.choice(np.arange(len(axis_angles)), replace=True, size=n_phi)
    axes_y = rng.choice(np.arange(len(axis_angles)), replace=True, size=n_phi)

    # compute the x and y components based on the axes
    phi_x = np.zeros((n_phi,))
    phi_y = np.zeros((n_phi,))

    for i in range(n_phi):
        aix = axes_x[i]
        if correlate_axes:
            aiy = axes_x[i]
        else:
            aiy = axes_y[i]
        if alternating_trig:
            if rng.choice([0, 1]) == 0:
                phi_x[i] = phi_list[i % len(phi_list)] * np.cos(axis_angles[aix] * np.pi / 180)
                phi_y[i] = phi_list[i % len(phi_list)] * np.sin(axis_angles[aiy] * np.pi / 180)
            else:
                phi_x[i] = phi_list[i % len(phi_list)] * np.sin(axis_angles[aix] * np.pi / 180)
                phi_y[i] = phi_list[i % len(phi_list)] * np.cos(axis_angles[aiy] * np.pi / 180)
        else:
            phi_x[i] = phi_list[i % len(phi_list)] * np.cos(axis_angles[aix] * np.pi / 180)
            phi_y[i] = phi_list[i % len(phi_list)] * np.sin(axis_angles[aiy] * np.pi / 180)

    #     assert np.all(np.abs(phi) >= np.pi * eps)
    #     assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fvx = np.zeros(dim, dtype='complex64')
    fvy = np.zeros(dim, dtype='complex64')

    fvx[0] = 1
    fvx[1:(dim + 1) // 2] = np.cos(phi_x) + 1j * np.sin(phi_x)
    fvx[-1:dim // 2:-1] = np.conj(fvx[1:(dim + 1) // 2])

    fvy[0] = 1
    fvy[1:(dim + 1) // 2] = np.cos(phi_y) + 1j * np.sin(phi_y)
    fvy[-1:dim // 2:-1] = np.conj(fvy[1:(dim + 1) // 2])

    if dim % 2 == 0:
        fvx[dim // 2] = 1
        fvy[dim // 2] = -1

    assert np.allclose(np.abs(fvx), 1)
    assert np.allclose(np.abs(fvy), 1)
    vx = np.fft.ifft(fvx)
    vy = np.fft.ifft(fvy)

    vx = vx.real
    vy = vy.real
    assert np.allclose(np.fft.fft(vx), fvx)
    assert np.allclose(np.linalg.norm(vx), 1)
    assert np.allclose(np.fft.fft(vy), fvy)
    assert np.allclose(np.linalg.norm(vy), 1)
    return vx, vy


class Visualizer(object):

    def __init__(self, cmap='plasma', vmin=None, vmax=None):

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

        self.cm = cm.get_cmap(cmap)

        self._nengo_html_ = ""

        # default starting config
        self.config = (13, 64, 32, 5, 5)

        self.apply_config()

    def apply_config(self):

        self.seed = self.config[0]
        self.dim = self.config[1]
        self.res = self.config[2]
        self.spacing = self.config[3]
        self.limits = self.config[4]
        self.xs = np.linspace(-self.limits, self.limits, self.res)
        self.ys = np.linspace(-self.limits, self.limits, self.res)
        self.x_axis_vec, self.y_axis_vec = make_periodic_axes(
            rng=np.random.RandomState(seed=self.seed),
            dim=self.dim,
            spacing=self.spacing,
            axis_angles=[0, 120, 240]
        )
        self.heatmap_vectors = get_heatmap_vectors(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_vec,
            y_axis_sp=self.y_axis_vec
        )

        # Origin point for similarity measures
        self.origin = np.zeros((self.dim,))
        self.origin[0] = 1

    def __call__(self, t, x):
        """
        seed
        dimensionality
        resolution
        spacing
        limits
        """

        seed = int(x[0])
        dimensionality = max(2, int(x[1]))
        resolution = max(16, int(x[2]))
        spacing = max(2, int(x[3]))
        # spacing = max(1, x[3])
        limits = max(.1, x[4])

        config = (seed, dimensionality, resolution, spacing, limits)

        # Only perform computation if something has changed since the last step
        if config != self.config:
            # Set the new config and update the visualization
            self.config = config

            self.apply_config()

            # Generate heatmap values
            self.vs = np.tensordot(self.origin, self.heatmap_vectors, axes=([0], [2]))

            if self.vmin is None:
                min_val = np.min(self.vs)
            else:
                min_val = self.vmin

            if self.vmax is None:
                max_val = np.max(self.vs)
            else:
                max_val = self.vmax

            self.vs = np.clip(self.vs, a_min=min_val, a_max=max_val)

            # xy = np.unravel_index(self.vs.argmax(), self.vs.shape)

            values = (self.cm(self.vs)*255).astype(np.uint8)

            self._nengo_html_ = image_svg(values)
