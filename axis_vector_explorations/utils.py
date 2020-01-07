import nengo.spa as spa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, \
    generate_region_vector, get_heatmap_vectors, spatial_dot
from spatial_semantic_pointers.plots import image_svg

# For the spatial spike html plot
from PIL import Image
import base64
import sys
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO

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


def angle_spacing_axes(ang_x, ang_y, off_x=0, off_y=0, dim=32):
    # X_test = ((np.arange(dim) * ang_x) + off_x) % (2 * np.pi)
    # Y_test = ((np.arange(dim) * ang_y) + off_y) % (2 * np.pi)
    X_test = ((np.arange(dim) * ang_x) + off_x) % 360
    Y_test = ((np.arange(dim) * ang_y) + off_y) % 360
    Xc = np.cos(X_test*np.pi/180) + 1j * np.sin(X_test*np.pi/180)
    Yc = np.cos(Y_test*np.pi/180) + 1j * np.sin(Y_test*np.pi/180)

    X = np.fft.ifft(Xc)
    Y = np.fft.ifft(Yc)

    X = spa.SemanticPointer(data=X)
    Y = spa.SemanticPointer(data=Y)
    X.make_unitary()
    Y.make_unitary()

    return X, Y


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


class AngleSpacingVisualizer(object):

    def __init__(self, cmap='plasma', vmin=None, vmax=None):

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

        self.cm = cm.get_cmap(cmap)

        self._nengo_html_ = ""

        # default starting config
        self.config = (13, 64, 32, 5, 30, 60, 0, 0)

        self.apply_config()

    def apply_config(self):

        self.seed = self.config[0]
        self.dim = self.config[1]
        self.res = self.config[2]
        self.limits = self.config[3]
        self.x_spacing = self.config[4]
        self.y_spacing = self.config[5]
        self.x_offset = self.config[6]
        self.y_offset = self.config[7]
        self.xs = np.linspace(-self.limits, self.limits, self.res)
        self.ys = np.linspace(-self.limits, self.limits, self.res)
        self.x_axis_vec, self.y_axis_vec = angle_spacing_axes(
            #rng=np.random.RandomState(seed=self.seed),
            ang_x=self.x_spacing,
            ang_y=self.y_spacing,
            off_x=self.x_offset,
            off_y=self.y_offset,
            dim=self.dim,
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

        seed = int(x[0])  # NOTE: this is currently unused
        dimensionality = max(2, int(x[1]))
        resolution = max(16, int(x[2]))
        limits = max(.1, x[3])

        # NOTE: these don't need to be int, just rounding to make things faster when scrolling through
        x_spacing = max(1, int(x[4]))
        y_spacing = max(1, int(x[5]))
        x_offset = int(x[6])
        y_offset = int(x[7])

        config = (seed, dimensionality, resolution, limits, x_spacing, y_spacing, x_offset, y_offset)

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


def image_svg(arr):
    """
    Given an array, return an svg image
    """
    if sys.version_info[0] == 3:
        # Python 3

        png = Image.fromarray(arr)
        buffer = BytesIO()
        png.save(buffer, format="PNG")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    else:
        # Python 2

        png = Image.fromarray(arr)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))


class CompleteSpatialSpikePlot(object):
    """
    Plots spiking activity of a neuron in relation to the spatial location of an agent
    Records activity of all neurons, and selects which one to show with an index input
    """

    def __init__(self, grid_size=512, n_neurons=500, xlim=(-1, 1), ylim=(-1, 1)):
        self.grid_size = grid_size
        self.xlim = xlim
        self.ylim = ylim
        self.space = np.zeros((n_neurons, grid_size, grid_size))
        self.x_len = xlim[1] - xlim[0]
        self.y_len = ylim[1] - ylim[0]
        self.n_neurons = n_neurons
        self._nengo_html_ = ''

    def __call__(self, t, x):

        x_loc = x[0]
        y_loc = x[1]
        index = int(x[2])
        spikes = x[3:]

        index = max(0, index)
        index = min(self.n_neurons - 1, index)

        # Get coordinates in the image
        """
        x_im = int((x_loc + self.x_len/2.)*1.*self.grid_size/self.x_len)
        #y_im = int((y_loc + self.y_len/2.)*1.*self.grid_size/self.y_len)
        y_im = int((-y_loc + self.y_len/2.)*1.*self.grid_size/self.y_len)
        """
        x_im = int((x_loc - self.xlim[0]) / self.x_len * self.grid_size)
        y_im = int((y_loc - self.ylim[0]) / self.y_len * self.grid_size)

        if x_im >= 0 and x_im < self.grid_size and y_im >= 0 and y_im < self.grid_size:

            # Place a spike or travelled path in the image
            for i, spike in enumerate(spikes):
                if spike > 0:
                    self.space[i, x_im, y_im] = 255
                    # self.space[i, y_im, x_im] = 255
                else:
                    if self.space[i, x_im, y_im] == 0:
                        self.space[i, x_im, y_im] = 128
                    # if self.space[i, y_im, x_im] == 0:
                    #    self.space[i, y_im, x_im] = 128

        # values = self.space[index,:,:].astype('uint8')
        values = self.space[index, :, :].astype('uint8').T
        self._nengo_html_ = image_svg(values)


#################################
# Trajectory generation classes #
#################################

class GeneratedOutput(object):
    def __init__(self, dt=0.001, vel_max=10, fixed_environment=True, xlim=(-3, 3), ylim=(-3, 3)):
        self.x = 0  # current x position
        self.y = 0  # current y position
        self.vx = 0  # current x velocity
        self.vy = 0  # current y velocity
        self.vel_max = vel_max  # maximum linear velocity
        self.ang = 0  # current facing angle
        self.vel = 0
        self.dt = dt

        self.xlim = xlim
        self.ylim = ylim
        self.fixed_environment = fixed_environment

    def step(self, t):
        raise NotImplemented

    def __call__(self, t):
        self.step(t)
        return self.x, self.y, self.vx, self.vy


class Spiral(GeneratedOutput):

    def __init__(self, ang_vel=2.9, lin_acc=.025, *args, **kwargs):

        # determines how tight the spiral is
        self.ang_vel = ang_vel

        # The linear acceleration, also determines tightness of spiral
        self.lin_acc = lin_acc

        GeneratedOutput.__init__(self, *args, **kwargs)

    def step(self, t):

        self.ang += self.ang_vel * self.dt

        self.vel += self.lin_acc * self.dt
        if self.vel > self.vel_max:
            self.vel = self.vel_max

        self.vx = self.vel*np.cos(self.ang)
        self.vy = self.vel*np.sin(self.ang)

        self.x += self.vx*self.dt
        self.y += self.vy*self.dt

        if self.fixed_environment:
            if self.x > self.xlim[1]:
                self.x = self.xlim[1]
            if self.x < self.xlim[0]:
                self.x = self.xlim[0]
            if self.y > self.ylim[1]:
                self.y = self.ylim[1]
            if self.y < self.ylim[0]:
                self.y = self.ylim[0]
        else:
            # Loop around if the environment is not fixed
            if self.x > 1:
                self.x -= 2
            if self.x < -1:
                self.x += 2
            if self.y > 1:
                self.y -= 2
            if self.y < -1:
                self.y += 2