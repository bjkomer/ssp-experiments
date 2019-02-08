from spatial_semantic_pointers.utils import generate_elliptic_region, ssp_to_loc
import numpy as np

class ExpandingNode(object):

    def __init__(self, current_loc_sp, goal_loc_sp, closest_landmark_id, allo_connections_sp, landmark_map_sp, landmark_vectors,
                 x_axis_sp, y_axis_sp, xs, ys, heatmap_vectors, diameter_increment=1):

        self.current_loc_sp = current_loc_sp
        self.goal_loc_sp = goal_loc_sp
        self.closest_landmark_id = closest_landmark_id
        self.allo_connections_sp = allo_connections_sp
        self.landmark_map_sp = landmark_map_sp
        self.landmark_vectors = landmark_vectors

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

        self.ellipse_sp = generate_elliptic_region(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_sp,
            y_axis_sp=self.y_axis_sp,
            f1=self.current_loc,
            f2=self.goal_loc,
            diameter=self.diameter,
        )

    def step(self):
        self.diameter += self.diameter_increment

        self.ellipse_sp = generate_elliptic_region(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_sp,
            y_axis_sp=self.y_axis_sp,
            f1=self.current_loc,
            f2=self.goal_loc,
            diameter=self.diameter,
        )

        potential_landmark = self.allo_connections_sp *~ self.ellipse_sp

        # Get closest vocab match that is above a threshold and that hasn't been found already
        #TODO
        if 'found':
            'add to list of already found landmarks'
            return 'landmark'
        else:
            return None


class EllipticExpansion(object):
    """
    Searches by creating an ellipse focussed on the current node and the goal node.
    Slowly expands the diameter of the ellipse until it encompasses a connection from the current node.
    When a connection is found, an ellipse starts expanding from that new node as well.
    This new node must remember which node triggered it,
    the triggering node will be considered its closest neighbor in the returned path
    All ellipses expand simultanously. Once a node connects to the goal, the search is complete
    """

    def __init__(self, start_landmark_id, end_landmark_id, landmark_map_sp, con_sp, landmark_vectors,
                 x_axis_sp, y_axis_sp, xs, ys, heatmap_vectors):

        self.start_landmark_id = start_landmark_id
        self.end_landmark_id = end_landmark_id
        self.landmark_map_sp = landmark_map_sp
        self.con_sp = con_sp
        self.landmark_vectors = landmark_vectors

        self.x_axis_sp = x_axis_sp
        self.y_axis_sp = y_axis_sp
        self.xs = xs
        self.ys = ys
        self.heatmap_vectors = heatmap_vectors

        current_loc_sp = self.landmark_map_sp * ~ self.start_landmark_id
        self.goal_loc_sp = self.landmark_map_sp * ~ self.end_landmark_id

        self.goal_loc = ssp_to_loc(
            self.goal_loc_sp,
            heatmap_vectors=self.heatmap_vectors,
            xs=self.xs,
            ys=self.ys
        )

        # egocentric displacements to nearby landmarks
        ego_connections_sp = con_sp * ~ start_landmark_id

        # allocentric coordinates of nearby landmarks
        allo_connections_sp = current_loc_sp * ego_connections_sp

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
                heatmap_vectors=self.heatmap_vectors
            )
        }

    def step(self):

        for key, val in self.expanding_nodes.iteritems():
            landmark_id = val.step()

            # Check to see if the goal is found
            if landmark_id == self.end_landmark_id:
                # goal is found, so build the path
                path = [landmark_id]

                cur_id = landmark_id

                while cur_id != self.start_landmark_id:
                    cur_id = self.expanding_nodes[cur_id].closest_landmark_id
                    path.append(cur_id)

                return path

            # If a new landmark is found, start expanding it
            elif landmark_id is not None and landmark_id not in self.expanding_nodes.keys():
                # location of the landmark in allocentric space
                current_loc_sp = self.landmark_map_sp * ~ landmark_id

                # egocentric displacements to nearby landmarks
                ego_connections_sp = self.con_sp * ~ landmark_id

                # allocentric coordinates of nearby landmarks
                allo_connections_sp = current_loc_sp * ego_connections_sp

                self.expanding_nodes[landmark_id] = ExpandingNode(
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
                    heatmap_vectors=self.heatmap_vectors
                )

        # The path wasn't found this iteration, so return None
        return None

    def find_path(self, max_steps=1000):

        for i in range(max_steps):

            ret = self.step()

            if ret is not None:
                return ret

        # No path found in the number of steps
        return None
