import numpy as np
from spatial_semantic_pointers.utils import power
from PIL import Image
import base64
import nengo
import nengo_spa as spa
from sklearn.metrics.pairwise import cosine_similarity
import sys
# Python version specific imports
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO


def orthogonal_hex_dir_7dim(phi=np.pi / 2., angle=0):
    dim = 7
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j * phi)
    xf[2] = 1
    xf[3] = 1
    xf[4] = np.conj(xf[3])
    xf[5] = np.conj(xf[2])
    xf[6] = np.conj(xf[1])

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = 1
    yf[2] = np.exp(1.j * phi)
    yf[3] = 1
    yf[4] = np.conj(yf[3])
    yf[5] = np.conj(yf[2])
    yf[6] = np.conj(yf[1])

    zf = np.zeros((dim,), dtype='Complex64')
    zf[0] = 1
    zf[1] = 1
    zf[2] = 1
    zf[3] = np.exp(1.j * phi)
    zf[4] = np.conj(zf[3])
    zf[5] = np.conj(zf[2])
    zf[6] = np.conj(zf[1])

    Xh = np.fft.ifft(xf).real
    Yh = np.fft.ifft(yf).real
    Zh = np.fft.ifft(zf).real

    # checks to make sure everything worked correctly
    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    assert np.allclose(np.fft.fft(Xh), xf)
    assert np.allclose(np.fft.fft(Yh), yf)
    assert np.allclose(np.linalg.norm(Xh), 1)
    assert np.allclose(np.linalg.norm(Yh), 1)

    axis_sps = [
        spa.SemanticPointer(data=Xh),
        spa.SemanticPointer(data=Yh),
        spa.SemanticPointer(data=Zh),
    ]

    n = 3
    points_nd = np.eye(n) * np.sqrt(n)
    # points in 2D that will correspond to each axis, plus one at zero
    points_2d = np.zeros((n, 2))
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1] + angle
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]

    X = power(axis_sps[0], x_axis[0])
    Y = power(axis_sps[0], y_axis[0])
    for i in range(1, n):
        X *= power(axis_sps[i], x_axis[i])
        Y *= power(axis_sps[i], y_axis[i])

    sv = transform_mat[3][0]
    return X, Y, sv


class EncoderPlot(nengo.Node):

    def __init__(self, connection, scaling='max'):

        self.connection = connection

        self.ensemble = connection.post_obj

        self.encoder_probe = nengo.Probe(connection.learning_rule, 'scaled_encoders')

        self.encoders = None

        self.scaling = scaling

        def plot(t):
            if self.encoders is None:
                return
            plot._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100">'
            #mx = (np.max(self.encoders) + np.mean(self.encoders))/2.
            if self.scaling == 'max':
                mx = np.max(self.encoders)
                if mx > 0:
                    self.encoders = self.encoders * (50. /mx)
            for e in self.encoders:
                if self.scaling == 'normalize':
                    e /= np.linalg.norm(e)
                    e *= 50
                plot._nengo_html_ += '<circle cx="{0}" cy="{1}" r="{2}"  stroke-width="1.0" stroke="blue" fill="blue" />'.format(e[0]+50, e[1]+50, 1)
            plot._nengo_html_ += '</svg>'

        super(EncoderPlot, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100"></svg>'

    def update(self, sim):
        if sim is None:
            return

        self.encoders = sim._probe_outputs[self.encoder_probe][-1]
        del sim._probe_outputs[self.encoder_probe][:]


class WeightPlot(nengo.Node):
    def __init__(self, connection, scaling='max'):

        self.connection = connection

        self.ensemble = connection.post_obj

        self.weight_probe = nengo.Probe(connection, 'weights', sample_every=0.01)

        self.weights = None

        self.scaling = scaling

        def plot(t):
            if self.weights is None:
                return
            mn = np.min(self.weights)
            mx = np.max(self.weights)
            rn = mx-mn
            self.weights = ((self.weights + mn)/rn)*255
            values = self.weights.astype('uint8')
            png = Image.fromarray(values)
            buffer = cStringIO.StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue())
            plot._nengo_html_ = '''
                <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.weights.shape[1], self.weights.shape[0])
            plot._nengo_html_ += '''
                <image width="100%%" height="100%%"
                       xlink:href="data:image/png;base64,%s"
                       style="image-rendering: pixelated;">
                </svg>''' % (''.join(img_str))

        super(WeightPlot, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100"></svg>'

    def update(self, sim):
        if sim is None:
            return
        try:
            self.weights = sim._probe_outputs[self.weight_probe][-1]
            del sim._probe_outputs[self.weight_probe][:]
        except:
            self.weights = None


class ItemMap(object):

    def __init__(self, shape, items, vocab, item_rad=.5,
                 start_x=1, start_y=1, start_th=0, dt=0.001,
                 motion_type='velocity'
                 ):

        self.shape = shape
        self.items = items  # dictionary of name:location
        self.vocab = vocab
        self.num_items = len(items)
        self.x = start_x
        self.y = start_y
        self.th = start_th
        self.item_rad = item_rad  # radius of item effect
        self.dt = dt

        # Scaling for positions on the HTML plot
        self.scale_x = 100. / self.shape[0]
        self.scale_y = 100. / self.shape[1]

        # Colours to display each item as
        self.colour_list = ['blue', 'green', 'red', 'magenta', 'cyan', 'yellow', 'purple', 'fuschia', 'grey', 'lime']
        self.num_colours = len(self.colour_list)

        self.build_html_string()

        self._nengo_html_ = self.base_html.format(self.x, self.y, self.th)

        if motion_type == 'velocity':
            self.move_func = self.move
        elif motion_type == 'random':
            self.move_func = self.move_random
        elif motion_type == 'teleport':
            self.move_func = self.teleport
        elif motion_type == 'holonomic':
            self.move_func = self.holonomic
        else:
            self.move_func = self.move

    def build_html_string(self):

        # Used to display HTML plot
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 100 100">'''

        # Draw the outer rectangle
        self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="white" />'

        # Draw circles for each item
        # for i, loc in enumerate(self.items.itervalues()):
        for i, loc in enumerate(self.items.values()):
            self.base_html += '<circle cx="{0}" cy="{1}" r="{2}" stroke-width="1.0" stroke="{3}" fill="{3}" />'.format(
                loc[0] * self.scale_x, 100 - loc[1] * self.scale_y, self.item_rad * self.scale_x,
                self.colour_list[i % self.num_colours])

        # Set up the agent to be filled in later with 'format()'
        self.base_html += '<polygon points="{0}" stroke="black" fill="black" />'

        # Close the svg
        self.base_html += '</svg>'

    def move(self, vel):

        self.th += vel[1] * self.dt
        if self.th > np.pi:
            self.th -= 2 * np.pi
        if self.th < -np.pi:
            self.th += 2 * np.pi

        self.x += np.cos(self.th) * vel[0] * self.dt
        self.y += np.sin(self.th) * vel[0] * self.dt

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()

    def holonomic(self, vel):

        self.x += vel[0] * self.dt
        self.y += vel[1] * self.dt

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()

    def teleport(self, pos):

        self.th = pos[2]
        if self.th > np.pi:
            self.th -= 2 * np.pi
        if self.th < -np.pi:
            self.th += 2 * np.pi

        self.x = pos[0]
        self.y = pos[1]

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()

    def move_random(self, dummy, vel=5):

        # randomly vary direction angle between -90 and +90 degrees
        # ang_diff = np.random.random() * np.pi - np.pi/2

        # randomly vary direction angle between -5 and +5 degrees
        ang_diff = (np.random.random() * np.pi - np.pi / 2) / 18

        self.th += ang_diff

        dx = vel * np.cos(self.th)
        dy = vel * np.sin(self.th)

        self.x += dx * self.dt
        self.y += dy * self.dt

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()

    def update_html(self, body_scale=0.5):
        # Define points of the triangular agent based on x, y, and th
        x1 = (self.x + body_scale * 0.5 * np.cos(self.th - 2 * np.pi / 3)) * self.scale_x
        y1 = 100 - (self.y + body_scale * 0.5 * np.sin(self.th - 2 * np.pi / 3)) * self.scale_y

        x2 = (self.x + body_scale * np.cos(self.th)) * self.scale_x
        y2 = 100 - (self.y + body_scale * np.sin(self.th)) * self.scale_y

        x3 = (self.x + body_scale * 0.5 * np.cos(self.th + 2 * np.pi / 3)) * self.scale_x
        y3 = 100 - (self.y + body_scale * 0.5 * np.sin(self.th + 2 * np.pi / 3)) * self.scale_y

        points = "{0},{1} {2},{3} {4},{5}".format(x1, y1, x2, y2, x3, y3)

        # Update the html plot
        self._nengo_html_ = self.base_html.format(points)

    def get_items(self):

        f = [0] * self.num_items
        # for i, loc in enumerate(self.items.itervalues()):
        for i, loc in enumerate(self.items.values()):
            f[i] = self.detect(loc)
        return f

    def detect(self, loc):

        dist = np.sqrt((self.x - loc[0]) ** 2 + (self.y - loc[1]) ** 2)
        if dist < self.item_rad:
            return 1
        else:
            return 0

    def generate_data(self, steps=1000, vel=5):
        """
        Generate artificial data corresponding to randomly moving through the environment
        """

        data = np.zeros((steps, 3 + self.num_items))

        for i in range(steps):
            self.move_random(vel=vel)

            f = self.get_items()
            data[i, :] = [self.x, self.y, self.th] + f

        return data

    def __call__(self, t, x):

        # self.move(x)
        # self.move_random()
        self.move_func(x)

        f = self.get_items()  # TODO: figure out the format of the items

        return [self.x, self.y, self.th] + f

class ExperimentControl(object):
    """
    Manages inputs to the network to facilitate an experiment
    Records relevant outputs as well
    """

    def __init__(self, items, vocab, file_name="exp_data_kosslyn", time_per_item=3, num_test_pairs=50):

        # Dictionary of {item: location}
        self.items = items

        self.keys = self.items.keys()
        # self.coords = self.items.values()  # Python 2
        self.coords = list(self.items.values())  # Python 3
        self.n_items = len(self.keys)
        self.item_index = -1  # Initialized to -1 because it increments right away
        self.prev_item_index = 0  # Keep track of the previous item for recording

        self.file_name = file_name

        # Data is stored in the form [start_item_index, end_item_index, elapsed_time, distance]
        self.data = np.zeros((num_test_pairs, 4))

        # Whether or not the time for the current trial has been recorded
        self.current_recorded = False

        self.phase = 'learning'

        # Copy of the item vocab used
        self.vocab = vocab
        self.vectors = self.vocab.vectors
        self.D = self.vectors.shape[1]
        # TODO: maybe just get the ordered vectors out of the vocab

        # Number of seconds to spend moving to each item during the learning phase
        # NOTE: current version takes about 2 seconds to go from one corner to another
        self.time_per_item = time_per_item

        # The number of pairs of items to mentally traverse between at test time
        self.num_test_pairs = num_test_pairs

        # Keep track of how many tests have currently been completed
        self.completed_tests = 0

        # TODO: get D from vocab
        # Construct the return vector here and index into it rather than concatenating other vectors
        self.ret_vec = np.zeros((2 + 3 + self.D))

        # The value of 't' when something last changed
        self.change_time = -self.time_per_item  # initialized negative so it triggers right away

    @staticmethod
    def vectors_close(v1, v2, threshold=0.85):
        """
        Returns true if the two vectors are close enough together, via cosine similarity
        """
        # return cosine_similarity(v1, v2) > threshold
        return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)) > threshold

    def __call__(self, t, x):
        """
        Returns [command, action, query_item]
        Dimensionality is 2 + 3 + D
        """

        if self.phase == 'learning':
            if t - self.change_time > self.time_per_item:
                self.change_time = t
                self.item_index += 1
                if self.item_index >= self.n_items:
                    # Learning complete, time to switch to the testing phase
                    self.phase = 'testing'
                    # Set the action to 'FIND'
                    self.ret_vec[[2, 3, 4]] = (0, 0, 1)
                    # Pick a random target item
                    # TODO: make the learning phase a random order so the test doesn't always start from the same item?
                    self.prev_item_index = self.item_index - 1
                    self.item_index = np.random.randint(low=0, high=self.n_items)
                else:
                    self.ret_vec[[0, 1]] = self.coords[self.item_index]
        elif self.phase == 'testing':
            # Choose random pairs to move between
            # For simplicity, use the last location as the starting location
            if t - self.change_time > self.time_per_item:
                # debugging
                # print(cosine_similarity(self.vectors[self.item_index], x))
                print(cosine_similarity(self.vectors[self.item_index].reshape(1, -1), x.reshape(1, -1)))

                self.current_recorded = False
                self.change_time = t
                self.completed_tests += 1
                if self.completed_tests >= self.num_test_pairs:
                    self.phase = 'done'
                    np.save(self.file_name, self.data)
                else:
                    next_item = np.random.randint(low=0, high=self.n_items)
                    # Keep choosing a random item to move to until it is different from the current item
                    while next_item == self.item_index:
                        next_item = np.random.randint(low=0, high=self.n_items)
                    self.prev_item_index = self.item_index
                    self.item_index = next_item

                    # Set the item vector output to be the target item
                    self.ret_vec[5:] = self.vectors[self.item_index]  # self.vocab.parse(self.keys[self.item_index]).v
            elif (not self.current_recorded) and self.vectors_close(self.vectors[self.item_index], x):
                # TODO: have a check here to see if the currently visualized item is close enough to the target item
                #       if so, record how long it took using 'self.change_time', and set some flag so that this doesn't
                #       get hit again until the next item. Could also move early, but the total time wouldn't be fixed
                elapsed_time = t - self.change_time

                dist = np.linalg.norm(self.coords[self.prev_item_index] - self.coords[self.item_index])
                # Store the timing data
                print("Storing timing data!")
                self.data[self.completed_tests, :] = (self.prev_item_index, self.item_index, elapsed_time, dist)
                self.current_recorded = True
        elif self.phase == 'done':
            # Done testing and recording, simulation should end now
            self.ret_vec[[2, 3, 4]] = (0, 0, 0)
            pass  # TODO: do something here?

        return self.ret_vec
