from spatial_semantic_pointers.utils import generate_elliptic_region_vector, ssp_to_loc
from spatial_semantic_pointers.plots import plot_heatmap, plot_vocab_similarity
import matplotlib.pyplot as plt
import numpy as np
import nengo.spa as spa


class ExpandingNode(object):

    def __init__(self, current_loc_sp, goal_loc_sp, closest_landmark_id, allo_connections_sp, landmark_map_sp, landmark_vectors,
                 x_axis_sp, y_axis_sp, xs, ys, heatmap_vectors, diameter_increment=1, expanded_list=[], threshold=0.08,
                 normalize=True
                 ):

        self.current_loc_sp = current_loc_sp
        self.goal_loc_sp = goal_loc_sp
        self.closest_landmark_id = closest_landmark_id
        self.allo_connections_sp = allo_connections_sp
        self.landmark_map_sp = landmark_map_sp
        self.landmark_vectors = landmark_vectors

        # List of the indices of already expanded nodes (as they appear in 'landmark_vectors'
        self.expanded_list = expanded_list

        # Similarity threshold for finding a match with the elliptic region
        # TODO: threshold should decrease with region size
        self.threshold = threshold

        # Whether or not to normalize the ellipse region SP
        self.normalize = normalize

        self.x_axis_sp = x_axis_sp
        self.y_axis_sp = y_axis_sp
        self.xs = xs
        self.ys = ys
        self.heatmap_vectors = heatmap_vectors

        # Amount to increase the diameter by on each step
        self.diameter_increment = diameter_increment

        self.current_loc = ssp_to_loc(
            self.current_loc_sp,
            heatmap_vectors=self.heatmap_vectors,
            xs=self.xs,
            ys=self.ys,
        )

        # TODO: this can just be calculated once rather than in every expanding node
        self.goal_loc = ssp_to_loc(
            self.goal_loc_sp,
            heatmap_vectors=self.heatmap_vectors,
            xs=self.xs,
            ys=self.ys,
        )

        # current diameter of the major axis of the ellipse
        # start it as the distance between the current node and the goal, forming a line
        self.diameter = np.linalg.norm(self.current_loc - self.goal_loc)

        # add some diameter to give it a width:
        self.diameter += self.diameter_increment

        self.ellipse_sp = generate_elliptic_region_vector(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_sp,
            y_axis_sp=self.y_axis_sp,
            f1=self.current_loc,
            f2=self.goal_loc,
            diameter=self.diameter,
            normalize=self.normalize,
        )

    def step(self):
        self.diameter += self.diameter_increment

        self.ellipse_sp = generate_elliptic_region_vector(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_sp,
            y_axis_sp=self.y_axis_sp,
            f1=self.current_loc,
            f2=self.goal_loc,
            diameter=self.diameter,
            normalize=self.normalize,
        )

        potential_landmark = self.allo_connections_sp * ~self.ellipse_sp

        # NOTE: this is only correct if allo_connections has landmark_id_sp information in it
        sim = np.tensordot(potential_landmark.v, self.landmark_vectors, axes=([0], [1]))

        # argsort sorts from lowest to highest, so create a view that reverses it
        inds = np.argsort(sim)[::-1]

        for i in inds:
            if sim[i] < self.threshold:
                # The next closest match is below threshold, so all others will be too
                return None
            elif i not in self.expanded_list:
                print("{} detected as closest connection with diameter {} and {} similarity".format(i, self.diameter, sim[i]))
                # Above threshold and has not been expanded yet, add it to the list now
                self.expanded_list.append(i)

                # Return the ID and the semantic pointer
                return i, spa.SemanticPointer(data=self.landmark_vectors[i])


class EllipticExpansion(object):
    """
    Searches by creating an ellipse focussed on the current node and the goal node.
    Slowly expands the diameter of the ellipse until it encompasses a connection from the current node.
    When a connection is found, an ellipse starts expanding from that new node as well.
    This new node must remember which node triggered it,
    the triggering node will be considered its closest neighbor in the returned path
    All ellipses expand simultanously. Once a node connects to the goal, the search is complete
    """

    def __init__(self, start_landmark_id, end_landmark_id, landmark_map_sp, con_ego_sp, con_allo_sp,
                 landmark_vectors, x_axis_sp, y_axis_sp, xs, ys, heatmap_vectors,
                 # params for debugging
                 true_allo_con_sps,
                 connectivity_list,
                 con_calculation='true_allo',
                 normalize=True,
                 debug_mode=False,
                 # ellipse params
                 diameter_increment=1,
                 **unused_params
                 ):

        self.debug_mode = debug_mode

        # Various methods for calculating the connectivity of a particular node. Used for debugging
        assert con_calculation in ['ego', 'allo', 'true_allo']
        self.con_calculation = con_calculation

        self.start_landmark_id = start_landmark_id
        self.end_landmark_id = end_landmark_id
        self.landmark_map_sp = landmark_map_sp
        self.con_ego_sp = con_ego_sp
        self.con_allo_sp = con_allo_sp
        self.landmark_vectors = landmark_vectors

        self.x_axis_sp = x_axis_sp
        self.y_axis_sp = y_axis_sp
        self.xs = xs
        self.ys = ys
        self.heatmap_vectors = heatmap_vectors

        self.true_allo_con_sps = true_allo_con_sps
        self.connectivity_list = connectivity_list
        # Whether or not to normalize the ellipse region SP
        self.normalize = normalize

        self.diameter_increment = diameter_increment

        start_landmark_sp = spa.SemanticPointer(self.landmark_vectors[self.start_landmark_id])
        end_landmark_sp = spa.SemanticPointer(self.landmark_vectors[self.end_landmark_id])

        current_loc_sp = self.landmark_map_sp * ~start_landmark_sp
        self.goal_loc_sp = self.landmark_map_sp * ~end_landmark_sp

        self.goal_loc = ssp_to_loc(
            self.goal_loc_sp,
            heatmap_vectors=self.heatmap_vectors,
            xs=self.xs,
            ys=self.ys
        )

        # egocentric displacements to nearby landmarks
        ego_connections_sp = con_ego_sp * ~start_landmark_sp

        # allocentric coordinates of nearby landmarks
        if self.con_calculation == 'ego':
            # calculating from ego
            allo_connections_sp = current_loc_sp * ego_connections_sp
        elif self.con_calculation == 'allo':
            # getting true value from allo
            allo_connections_sp = self.con_allo_sp * ~start_landmark_sp
        elif self.con_calculation == 'true_allo':
            # get a clean value from allo
            allo_connections_sp = self.true_allo_con_sps[self.start_landmark_id]
        else:
            raise NotImplementedError

        # dictionary of nodes currently being expanded
        self.expanding_nodes = {
            self.start_landmark_id: ExpandingNode(
                current_loc_sp=current_loc_sp,
                goal_loc_sp=self.goal_loc_sp,
                closest_landmark_id=self.start_landmark_id,
                allo_connections_sp=allo_connections_sp,
                landmark_map_sp=self.landmark_map_sp,
                landmark_vectors=self.landmark_vectors,
                x_axis_sp=self.x_axis_sp,
                y_axis_sp=self.y_axis_sp,
                xs=self.xs,
                ys=self.ys,
                heatmap_vectors=self.heatmap_vectors,
                normalize=self.normalize,
            )
        }

    def step(self):

        # dictionary is modified in loop, so get a snapshot of the keys before iterating
        for key in list(self.expanding_nodes.keys()):
            # Will be None, or a tuple of (id, sp)
            landmark = self.expanding_nodes[key].step()

            # Nothing found this step, so skip it
            if landmark is None:
                continue

            # Check to see if the goal is found
            # if np.allclose(landmark_id.v, self.end_landmark_id.v):
            elif landmark[0] == self.end_landmark_id:
                if self.debug_mode:
                    print("Goal is found, connecting {} to {}".format(key, landmark[0]))
                    print(self.expanding_nodes.keys())
                # goal is found, so build the path
                path = [landmark[0], key]

                cur_id = key#landmark[0] #FIXME

                while cur_id != self.start_landmark_id:
                    cur_id = self.expanding_nodes[cur_id].closest_landmark_id
                    path.append(cur_id)

                return path

            # If a new landmark is found, start expanding it
            elif landmark[0] not in self.expanding_nodes.keys():
                if self.debug_mode:
                    print("Found a new landmark, connecting {} to {}".format(key, landmark[0]))
                # location of the landmark in allocentric space
                current_loc_sp = self.landmark_map_sp * ~landmark[1]

                # egocentric displacements to nearby landmarks
                ego_connections_sp = self.con_ego_sp * ~landmark[1]

                # allocentric coordinates of nearby landmarks
                if self.con_calculation == 'ego':
                    # calculating from ego
                    allo_connections_sp = current_loc_sp * ego_connections_sp
                elif self.con_calculation == 'allo':
                    # getting true value from allo
                    allo_connections_sp = self.con_allo_sp * ~landmark[1]
                elif self.con_calculation == 'true_allo':
                    # get a clean value from allo
                    allo_connections_sp = self.true_allo_con_sps[landmark[0]]
                else:
                    raise NotImplementedError

                self.expanding_nodes[landmark[0]] = ExpandingNode(
                    current_loc_sp=current_loc_sp,
                    goal_loc_sp=self.goal_loc_sp,
                    closest_landmark_id=key,
                    allo_connections_sp=allo_connections_sp,
                    landmark_map_sp=self.landmark_map_sp,
                    landmark_vectors=self.landmark_vectors,
                    x_axis_sp=self.x_axis_sp,
                    y_axis_sp=self.y_axis_sp,
                    xs=self.xs,
                    ys=self.ys,
                    heatmap_vectors=self.heatmap_vectors,
                    normalize=self.normalize,
                    diameter_increment=self.diameter_increment,
                )

        # The path wasn't found this iteration, so return None
        return None

    def find_path(self, max_steps=1000, display=True, **display_params):

        if display:
            fig, ax = plt.subplots(3, 7)
            plt.ion()
            plt.show()

        for i in range(max_steps):
            if self.debug_mode:
                print("Step {0} of {1}".format(i + 1, max_steps))

            ret = self.step()

            if display:
                ax[0, 0].cla()
                self.display_search(
                    ax=ax[0, 0], true_graph=display_params['graph'],
                    xs=display_params['xs'], ys=display_params['ys'],
                )

                # Plot the heatmaps of each ellipse SP
                for key, value in self.expanding_nodes.items():
                    plot_heatmap(
                        value.ellipse_sp.v,
                        display_params['heatmap_vectors'],
                        ax[1, key],
                        display_params['xs'],
                        display_params['ys'],
                        name='',
                        vmin=-1,
                        vmax=1,
                        # vmin=None,
                        # vmax=None,
                        cmap='plasma'
                    )

                    plot_vocab_similarity(
                        vec=(value.allo_connections_sp *~ value.ellipse_sp).v,
                        vocab_vectors=self.landmark_vectors,
                        ax=ax[2, key],
                    )

                    connections = np.array(self.connectivity_list[key])

                    # Plot a dot on the bars that correspond to landmarks that should be connected
                    # These bars should be higher than the rest if the output is correct
                    # Note that they should only be high if the ellipse overlaps the node, which often isn't the case
                    # for backward connections. This is desirable as backwards connections are likely not the shortest
                    ax[2, key].scatter(x=connections, y=np.zeros((len(connections))))

                plt.draw()
                plt.pause(0.001)
                input("Press [enter] to continue.")

            if ret is not None:
                if display:
                    print("Path found: ", ret[::-1])
                    input("Press [enter] to exit.")
                # Reverse the found path to be from start to goal rather than goal to start
                return ret[::-1]

        # No path found in the number of steps
        return None

    def display_search(self, ax, true_graph, xs, ys):
        """
        Displays what the search looks like so far
        :return:
        """

        # Plot the nodes and their connectivity
        true_graph.plot_graph(ax=ax, xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))

        # Plot the current ellipses

        f2 = true_graph.nodes[6].data['location']
        for key, value in self.expanding_nodes.items():
            # x, y = self.ellipse_points(f1=value.current_loc, f2=value.goal_loc, diameter=value.diameter)
            f1 = true_graph.nodes[key].data['location']
            x, y = self.ellipse_points(f1=f1, f2=f2, diameter=value.diameter)
            ax.plot(x, y)

    @staticmethod
    def ellipse_points(f1, f2, diameter):

        # https://stackoverflow.com/a/42266310/1189797

        # Compute ellipse parameters
        a = diameter / 2                                     # Semimajor axis
        x0 = (f1[0] + f2[0]) / 2                             # Center x-value
        y0 = (f1[1] + f2[1]) / 2                             # Center y-value
        f = np.sqrt((f1[0] - x0)**2 + (f1[1] - y0)**2)       # Distance from center to focus
        b = np.sqrt(a**2 - f**2)                             # Semiminor axis
        phi = np.arctan2((f2[1] - f1[1]), (f2[0] - f1[0]))   # Angle between major axis and x-axis

        # Parametric plot in t
        resolution = 1000
        t = np.linspace(0, 2*np.pi, resolution)
        x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)

        return x, y


