import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ItemMap(object):

    def __init__(self, shape, items, vocab, item_rad=.5,
                 start_x=1, start_y=1, start_th=0, dt=0.001,
                 motion_type='holonomic'
                 ):

        self.shape = shape
        self.items = items  # dictionary of name:location
        self.item_names = list(self.items.keys())
        self.vocab = vocab
        self.num_items = len(items)
        self.x = start_x
        self.y = start_y
        self.th = start_th
        self.item_rad = item_rad  # radius of item effect
        self.dt = dt

        self.dim = self.vocab.dimensions

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

        f = np.zeros((self.dim, ))
        # for i, loc in enumerate(self.items.itervalues()):
        for i, loc in enumerate(self.items.values()):
            if self.detect(loc):
                f += self.vocab[self.item_names[i]].v
        return f

    def detect(self, loc):

        dist = np.sqrt((self.x - loc[0]) ** 2 + (self.y - loc[1]) ** 2)
        return dist < self.item_rad

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

        return [self.x, self.y] + list(f)


class ExperimentControl(object):
    """
    Manages inputs to the network to facilitate an experiment
    Records relevant outputs as well
    """

    def __init__(self, items, vocab, file_name, time_per_item=3, num_test_pairs=50, sim_thresh=.8,
                 dir_mag_limit=1.0, ssp_dir=True, encode_func=None, decode_func=None):

        # Dictionary of {item: location}
        self.items = items

        self.keys = list(self.items.keys())
        # self.coords = self.items.values()  # Python 2
        self.coords = list(self.items.values())  # Python 3
        self.n_items = len(self.keys)
        self.item_index = -1  # Initialized to -1 because it increments right away
        self.prev_item_index = 0  # Keep track of the previous item for recording

        # output direction to move as an ssp
        self.ssp_dir = ssp_dir

        # encoding and decoding to SSP, if required
        self.encode_func = encode_func
        self.decode_func = decode_func

        self.file_name = file_name

        # Data is stored in the form [start_item_index, end_item_index, elapsed_time, distance]
        self.data = np.zeros((num_test_pairs, 4))

        # Whether or not the time for the current trial has been recorded
        self.current_recorded = False

        self.phase = 'learning'

        # threshold for similarity
        self.sim_thresh = sim_thresh
        # 'velocity' of the circular convolution
        self.dir_mag_limit = dir_mag_limit

        # Copy of the item vocab used
        self.vocab = vocab
        self.vectors = self.vocab.vectors
        self.dim = self.vectors.shape[1]
        # TODO: maybe just get the ordered vectors out of the vocab

        # Number of seconds to spend moving to each item during the learning phase
        # NOTE: current version takes about 2 seconds to go from one corner to another
        self.time_per_item = time_per_item

        # The number of pairs of items to mentally traverse between at test time
        self.num_test_pairs = num_test_pairs

        # Keep track of how many tests have currently been completed
        self.completed_tests = 0

        # Construct the return vector here and index into it rather than concatenating other vectors
        if self.ssp_dir:
            # also includes an input to the memory, to initialize it at the correct location
            self.ret_vec = np.zeros((3 * self.dim,))
            # initialize to the origin (movement and visualized location)
            self.ret_vec[2 * self.dim] = 1
            self.ret_vec[self.dim] = 1
        else:
            self.ret_vec = np.zeros((2 + self.dim,))

        # The value of 't' when something last changed
        self.change_time = -self.time_per_item  # initialized negative so it triggers right away

        self.done = False

        # variables for the html plot
        self.x = 0
        self.y = 0
        self.th = 0
        # Scaling for positions on the HTML plot
        self.scale_x = 100. / 10
        self.scale_y = 100. / 10
        self.item_rad = 0.5

        # Colours to display each item as
        self.colour_list = ['blue', 'green', 'red', 'magenta', 'cyan', 'yellow', 'purple', 'fuschia', 'grey', 'lime']
        self.num_colours = len(self.colour_list)

        self.build_html_string()

        self._nengo_html_ = self.base_html.format(self.x, self.y, self.th)

    @staticmethod
    def vectors_close(v1, v2, threshold=0.85):
        """
        Returns true if the two vectors are close enough together, via cosine similarity
        """
        # return cosine_similarity(v1, v2) > threshold
        return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)) > threshold

    def build_html_string(self):

        # Used to display HTML plot
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 100 100">'''

        # Draw the outer rectangle
        self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="white" />'

        # Draw circles for each item
        # for i, loc in enumerate(self.items.itervalues()):
        for i, loc in enumerate(self.items.values()):
            self.base_html += '<circle cx="{0}" cy="{1}" r="{2}" stroke-width="1.0" stroke="{3}" fill="{3}" />'.format(
                (loc[0]+5) * self.scale_x, 100 - (loc[1]+5) * self.scale_y, self.item_rad * self.scale_x,
                self.colour_list[i % self.num_colours])

        # Set up the agent to be filled in later with 'format()'
        self.base_html += '<polygon points="{0}" stroke="black" fill="black" />'

        # Close the svg
        self.base_html += '</svg>'


    def update_html(self, body_scale=0.5):
        # Define points of the triangular agent based on x, y, and th
        x1 = (self.x+5 + body_scale * 0.5 * np.cos(self.th - 2 * np.pi / 3)) * self.scale_x
        y1 = 100 - (self.y+5 + body_scale * 0.5 * np.sin(self.th - 2 * np.pi / 3)) * self.scale_y

        x2 = (self.x+5 + body_scale * np.cos(self.th)) * self.scale_x
        y2 = 100 - (self.y+5 + body_scale * np.sin(self.th)) * self.scale_y

        x3 = (self.x+5 + body_scale * 0.5 * np.cos(self.th + 2 * np.pi / 3)) * self.scale_x
        y3 = 100 - (self.y+5 + body_scale * 0.5 * np.sin(self.th + 2 * np.pi / 3)) * self.scale_y

        points = "{0},{1} {2},{3} {4},{5}".format(x1, y1, x2, y2, x3, y3)

        # Update the html plot
        self._nengo_html_ = self.base_html.format(points)

    def __call__(self, t, x):
        """
        input x is [current_coord, mental_vision]
        Returns [direction, query_item]
        Dimensionality is 2 + dim
        """

        if not self.done:
            if self.ssp_dir:
                cur_coord = x[:self.dim]
                vis = x[self.dim:]

                # set the direction to move to get there
                estim = self.decode_func(cur_coord)
                # this is for updating the html plot
                self.x = estim[0]
                self.y = estim[1]
                self.update_html()
                displacement = self.coords[self.item_index] - estim
                mag = np.linalg.norm(displacement)
                # only normalize if far away, so when super close, it won't bounce around
                # 0.25 should be within hitting distance
                if mag > 0.25:
                    displacement = displacement / mag
                    displacement *= self.dir_mag_limit
                self.ret_vec[:self.dim] = self.encode_func(displacement)
            else:
                cur_coord = x[:2]
                vis = x[2:]

                # update desired direction based on any drift
                # get the unit vector distance in the desired direction
                displacement = self.coords[self.item_index] - cur_coord  # self.coords[self.prev_item_index]
                displacement = displacement / np.linalg.norm(displacement)

                # set the direction to move to get there
                self.ret_vec[:2] = displacement

            # Choose random pairs to move between
            # For simplicity, use the last location as the starting location
            if t - self.change_time > self.time_per_item:
                # debugging
                # print(cosine_similarity(self.vectors[self.item_index], x))
                # print(cosine_similarity(self.vectors[self.item_index].reshape(1, -1), x.reshape(1, -1)))

                self.current_recorded = False
                self.change_time = t
                self.completed_tests += 1
                if self.completed_tests >= self.num_test_pairs:
                    self.done = True
                else:
                    next_item = np.random.randint(low=0, high=self.n_items)
                    # Keep choosing a random item to move to until it is different from the current item
                    while next_item == self.item_index:
                        next_item = np.random.randint(low=0, high=self.n_items)
                    self.prev_item_index = self.item_index
                    self.item_index = next_item

                    if self.ssp_dir:
                        # Set the item vector output to be the target item
                        self.ret_vec[self.dim:2*self.dim] = self.vectors[self.item_index]  # self.vocab.parse(self.keys[self.item_index]).v

                        displacement = self.coords[self.item_index] - self.decode_func(cur_coord)
                        displacement = displacement / np.linalg.norm(displacement)
                        # set the direction to move to get there
                        self.ret_vec[:self.dim] = self.encode_func(displacement)

                        # stop driving the memory to the start location
                        self.ret_vec[2 * self.dim:] = 0
                    else:
                        # Set the item vector output to be the target item
                        self.ret_vec[2:] = self.vectors[
                            self.item_index]  # self.vocab.parse(self.keys[self.item_index]).v

                        # get the unit vector distance in the desired direction
                        displacement = self.coords[self.item_index] - cur_coord  # self.coords[self.prev_item_index]
                        displacement = displacement / np.linalg.norm(displacement)

                        # set the direction to move to get there
                        self.ret_vec[:2] = displacement
            # elif (not self.current_recorded) and self.vectors_close(self.vectors[self.item_index], vis):
            elif (not self.current_recorded) and np.dot(self.vectors[self.item_index], vis) > self.sim_thresh:
                # TODO: have a check here to see if the currently visualized item is close enough to the target item
                #       if so, record how long it took using 'self.change_time', and set some flag so that this doesn't
                #       get hit again until the next item. Could also move early, but the total time wouldn't be fixed
                elapsed_time = t - self.change_time

                dist = np.linalg.norm(self.coords[self.prev_item_index] - self.coords[self.item_index])
                # Store the timing data
                print("Storing timing data!")
                self.data[self.completed_tests, :] = (self.prev_item_index, self.item_index, elapsed_time, dist)
                # saving data so far, overwriting old file
                np.save(self.file_name, self.data[:self.completed_tests])
                self.current_recorded = True

                if self.ssp_dir:
                    # at target, don't need to move (setting displacement to origin)
                    self.ret_vec[:self.dim] = 0
                    self.ret_vec[0] = 1
                    # modify the memory to start at the last target
                    self.ret_vec[2*self.dim:] = self.encode_func(self.coords[self.item_index])
                else:
                    # at target, don't need to move
                    self.ret_vec[:2] = 0
            # else:
            #     print(np.dot(self.vectors[self.item_index], vis))

        return self.ret_vec


# class OldExperimentControl(object):
#     """
#     Manages inputs to the network to facilitate an experiment
#     Records relevant outputs as well
#     """
#
#     def __init__(self, items, vocab, file_name="exp_data_kosslyn", time_per_item=3, num_test_pairs=50):
#
#         # Dictionary of {item: location}
#         self.items = items
#
#         self.keys = self.items.keys()
#         # self.coords = self.items.values()  # Python 2
#         self.coords = list(self.items.values())  # Python 3
#         self.n_items = len(self.keys)
#         self.item_index = -1  # Initialized to -1 because it increments right away
#         self.prev_item_index = 0  # Keep track of the previous item for recording
#
#         self.file_name = file_name
#
#         # Data is stored in the form [start_item_index, end_item_index, elapsed_time, distance]
#         self.data = np.zeros((num_test_pairs, 4))
#
#         # Whether or not the time for the current trial has been recorded
#         self.current_recorded = False
#
#         self.phase = 'learning'
#
#         # Copy of the item vocab used
#         self.vocab = vocab
#         self.vectors = self.vocab.vectors
#         self.D = self.vectors.shape[1]
#         # TODO: maybe just get the ordered vectors out of the vocab
#
#         # Number of seconds to spend moving to each item during the learning phase
#         # NOTE: current version takes about 2 seconds to go from one corner to another
#         self.time_per_item = time_per_item
#
#         # The number of pairs of items to mentally traverse between at test time
#         self.num_test_pairs = num_test_pairs
#
#         # Keep track of how many tests have currently been completed
#         self.completed_tests = 0
#
#         # TODO: get D from vocab
#         # Construct the return vector here and index into it rather than concatenating other vectors
#         self.ret_vec = np.zeros((2 + 3 + self.D))
#
#         # The value of 't' when something last changed
#         self.change_time = -self.time_per_item  # initialized negative so it triggers right away
#
#     @staticmethod
#     def vectors_close(v1, v2, threshold=0.85):
#         """
#         Returns true if the two vectors are close enough together, via cosine similarity
#         """
#         # return cosine_similarity(v1, v2) > threshold
#         return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)) > threshold
#
#     def __call__(self, t, x):
#         """
#         Returns [command, action, query_item]
#         Dimensionality is 2 + 3 + D
#         """
#
#         if self.phase == 'learning':
#             if t - self.change_time > self.time_per_item:
#                 self.change_time = t
#                 self.item_index += 1
#                 if self.item_index >= self.n_items:
#                     # Learning complete, time to switch to the testing phase
#                     self.phase = 'testing'
#                     # Set the action to 'FIND'
#                     self.ret_vec[[2, 3, 4]] = (0, 0, 1)
#                     # Pick a random target item
#                     # TODO: make the learning phase a random order so the test doesn't always start from the same item?
#                     self.prev_item_index = self.item_index - 1
#                     self.item_index = np.random.randint(low=0, high=self.n_items)
#                 else:
#                     self.ret_vec[[0, 1]] = self.coords[self.item_index]
#         elif self.phase == 'testing':
#             # Choose random pairs to move between
#             # For simplicity, use the last location as the starting location
#             if t - self.change_time > self.time_per_item:
#                 # debugging
#                 # print(cosine_similarity(self.vectors[self.item_index], x))
#                 print(cosine_similarity(self.vectors[self.item_index].reshape(1, -1), x.reshape(1, -1)))
#
#                 self.current_recorded = False
#                 self.change_time = t
#                 self.completed_tests += 1
#                 if self.completed_tests >= self.num_test_pairs:
#                     self.phase = 'done'
#                     np.save(self.file_name, self.data)
#                 else:
#                     next_item = np.random.randint(low=0, high=self.n_items)
#                     # Keep choosing a random item to move to until it is different from the current item
#                     while next_item == self.item_index:
#                         next_item = np.random.randint(low=0, high=self.n_items)
#                     self.prev_item_index = self.item_index
#                     self.item_index = next_item
#
#                     # Set the item vector output to be the target item
#                     self.ret_vec[5:] = self.vectors[self.item_index]  # self.vocab.parse(self.keys[self.item_index]).v
#             elif (not self.current_recorded) and self.vectors_close(self.vectors[self.item_index], x):
#                 # TODO: have a check here to see if the currently visualized item is close enough to the target item
#                 #       if so, record how long it took using 'self.change_time', and set some flag so that this doesn't
#                 #       get hit again until the next item. Could also move early, but the total time wouldn't be fixed
#                 elapsed_time = t - self.change_time
#
#                 dist = np.linalg.norm(self.coords[self.prev_item_index] - self.coords[self.item_index])
#                 # Store the timing data
#                 print("Storing timing data!")
#                 self.data[self.completed_tests, :] = (self.prev_item_index, self.item_index, elapsed_time, dist)
#                 self.current_recorded = True
#         elif self.phase == 'done':
#             # Done testing and recording, simulation should end now
#             self.ret_vec[[2, 3, 4]] = (0, 0, 0)
#             pass  # TODO: do something here?
#
#         return self.ret_vec
