from spatial_semantic_pointers.utils import generate_elliptic_region_vector, ssp_to_loc
import numpy as np
import nengo.spa as spa


class ExpandingNode(object):

    def __init__(self, current_loc_sp, goal_loc_sp, closest_landmark_id, allo_connections_sp, landmark_map_sp, landmark_vectors,
                 x_axis_sp, y_axis_sp, xs, ys, heatmap_vectors, diameter_increment=1, expanded_list=[], threshold=0.1):

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

        self.ellipse_sp = generate_elliptic_region_vector(
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

        self.ellipse_sp = generate_elliptic_region_vector(
            xs=self.xs,
            ys=self.ys,
            x_axis_sp=self.x_axis_sp,
            y_axis_sp=self.y_axis_sp,
            f1=self.current_loc,
            f2=self.goal_loc,
            diameter=self.diameter,
        )

        potential_landmark = self.allo_connections_sp *~ self.ellipse_sp


        sim = np.tensordot(potential_landmark.v, self.landmark_vectors, axes=([0], [1]))

        # argsort sorts from lowest to highest, so create a view that reverses it
        inds = np.argsort(sim)[::-1]

        for i in inds:
            if sim[i] < self.threshold:
                # The next closest match is below threshold, so all others will be too
                return None
            elif i not in self.expanded_list:
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

        start_landmark_sp = spa.SemanticPointer(self.landmark_vectors[self.start_landmark_id])
        end_landmark_sp = spa.SemanticPointer(self.landmark_vectors[self.end_landmark_id])

        current_loc_sp = self.landmark_map_sp * ~ start_landmark_sp
        self.goal_loc_sp = self.landmark_map_sp * ~ end_landmark_sp

        self.goal_loc = ssp_to_loc(
            self.goal_loc_sp,
            heatmap_vectors=self.heatmap_vectors,
            xs=self.xs,
            ys=self.ys
        )

        # egocentric displacements to nearby landmarks
        ego_connections_sp = con_sp * ~ start_landmark_sp

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

        # dictionary is modified in loop, so get a snapshot of the keys before iterating
        for key in list(self.expanding_nodes.keys()):
            # Will be None, or a tuple of (id, sp)
            landmark = self.expanding_nodes[key].step()

            # Check to see if the goal is found
            # if np.allclose(landmark_id.v, self.end_landmark_id.v):
            if landmark[0] == self.end_landmark_id:
                print("Goal is found")
                print(self.expanding_nodes.keys())
                # goal is found, so build the path
                path = [landmark[0]]

                cur_id = key#landmark[0] #FIXME

                while cur_id != self.start_landmark_id:
                    cur_id = self.expanding_nodes[cur_id].closest_landmark_id
                    path.append(cur_id)

                return path

            # If a new landmark is found, start expanding it
            elif landmark is not None and landmark[0] not in self.expanding_nodes.keys():
                print("Found a new landmark")
                # location of the landmark in allocentric space
                current_loc_sp = self.landmark_map_sp * ~ landmark[1]

                # egocentric displacements to nearby landmarks
                ego_connections_sp = self.con_sp * ~ landmark[1]

                # allocentric coordinates of nearby landmarks
                allo_connections_sp = current_loc_sp * ego_connections_sp

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
                    heatmap_vectors=self.heatmap_vectors
                )

        # The path wasn't found this iteration, so return None
        return None

    def find_path(self, max_steps=1000):

        for i in range(max_steps):
            print("Step {0} of {1}".format(i + 1, max_steps))

            ret = self.step()

            if ret is not None:
                return ret

        # No path found in the number of steps
        return None
