import numpy as np
from skimage.draw import line, line_aa


class Graph(object):

    def __init__(self, name='graph', matrix=None, data=None):

        self.name = name

        self.nodes = []

        self.n_nodes = 0

        if matrix is not None:
            self.construct_from_matrix(matrix, data)

    def construct_from_matrix(self, matrix, data=None):
        """
        Construct a graph based on connectivity defined in a matrix
        0 means the two nodes are not connected
        any number represents the connections strength
        data is an option list of information to go with each node
        """

        assert matrix.shape[0] == matrix.shape[1]

        self.n_nodes = matrix.shape[0]

        # Add all the nodes to the graphs
        for i in range(self.n_nodes):
            if data is not None:
                self.nodes.append(Node(index=i, data=data[i]))
            else:
                self.nodes.append(Node(index=i))

            # Set the appropriate neighbors for the node
            for j in range(self.n_nodes):
                if matrix[i, j] > 0:
                    self.nodes[i].add_neighbor(j, matrix[i, j])

    def search_graph(self, start_node, end_node):
        """
        Returns the path from start_node to end_node as a list
        :param start_node: index of the start node
        :param end_node: index of the end node
        :return: list of nodes along the path
        """

        # Keep track of shortest distances to each node (from the end_node)
        distances = 1000 * np.ones((self.n_nodes,))

        distances[end_node] = 0

        to_expand = [end_node]

        finished = False

        while len(to_expand) > 0:# and not finished:
            next_node = to_expand.pop()

            for neighbor, dist in self.nodes[next_node].neighbors.items():
                # If a faster way to get to the neighbor is detected, update the distances with the
                # new shorter distance, and set this neighbor to be expanded
                if distances[next_node] + dist < distances[neighbor]:
                    distances[neighbor] = distances[next_node] + dist
                    to_expand.append(neighbor)

        # Now follow the shortest path from start to goal
        path = []

        # If the start and goal cannot be connected, return an empty list
        if distances[start_node] >= 1000:
            return []

        current_node = start_node

        while current_node != end_node:
            path.append(current_node)
            # choose the node connected to the current node with the shortest distance to the goal
            neighbors = self.nodes[current_node].get_neighbors()

            closest_n = -1
            closest_dist = 1000
            for n in neighbors:
                # print("neighbor: {0}".format(n))
                if distances[n] < closest_dist:
                    closest_dist = distances[n]
                    closest_n = n

            current_node = closest_n
            assert current_node != -1

        path.append(current_node)

        return path

    def search_graph_by_location(self, start_location, end_location):

        start_node = self.get_node_for_location(start_location)
        end_node = self.get_node_for_location(end_location)

        # self.search_graph() gives a list of indices, need to convert those to x-y locations before returning
        return [self.nodes[n].data['location'] for n in self.search_graph(start_node.index, end_node.index)]

    def get_node_for_location(self, location):

        closest_dist = 10000
        closest_node = -1
        for n in self.nodes:
            if np.all(n.data['location'] == location):
                # If there is an exact match, return it now
                return n
            else:
                # If it's not a match, see if it is the closest match
                dist = np.linalg.norm(n.data['location'] - location)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_node = n

        # No match found, return the closest node
        # Due to rounding there will almost never be an exact match, so the warning message is not helpful
        # print("Warning: no matching node location found, using closest location of {0} for {1}".format(
        #     closest_node.data['location'], location
        # ))
        return closest_node

    def is_valid_path(self, path):
        """
        Given a sequence of node indices, return true if they represent a valid path in the graph and false otherwise
        :param path: list of node indices
        :return: boolean
        """

        for i in range(len(path) - 1):
            if path[i + 1] not in self.nodes[path[i]].get_neighbors():
                return False

        return True


    def plot_graph(self, ax, xlim=(0, 10), ylim=(0, 10), invert=False):
        """
        plots the graph structure of connected nodes on an a pyplot axis
        assumes that each node has a x-y location in its data
        """

        ax.cla()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        for n in self.nodes:
            x = n.data['location'][0]
            y = n.data['location'][1]

            # Draw a dot at the node location
            if invert:
                # Invert y for plotting
                # Flip x and y axes for plotting
                ax.scatter(y, ylim[1] - x)
            else:
                ax.scatter(x, y)

            # Go through the neighbor indices of each neighbor
            for ni in n.get_neighbors():
                # Draw a line between the current node and the neighbor node
                nx = self.nodes[ni].data['location'][0]
                ny = self.nodes[ni].data['location'][1]
                if invert:
                    # Invert y for plotting
                    # Flip x and y axes for plotting
                    ax.plot([y, ny], [ylim[1] - x, ylim[1] - nx])
                else:
                    ax.plot([x, nx], [y, ny])

    def graph_image(self, xs, ys):
        """
        creates an image that represents the graph structure of connected nodes
        used for plotting as an image, so that the scale is the same as the occupancy images
        """

        img = np.zeros((len(xs), len(ys)))

        for n in self.nodes:
            x = n.data['location'][0]
            y = n.data['location'][1]

            # Index of the coordinate in the array
            # NOTE: this might be slightly incorrect....?
            ix = (np.abs(xs - x)).argmin()
            iy = (np.abs(ys - y)).argmin()

            img[ix, iy] = 1  #TEMP

            # Go through the neighbor indices of each neighbor
            for ni in n.get_neighbors():
                # Draw a line between the current node and the neighbor node
                nx = self.nodes[ni].data['location'][0]
                ny = self.nodes[ni].data['location'][1]

                inx = (np.abs(xs - nx)).argmin()
                iny = (np.abs(ys - ny)).argmin()

                # rr, cc = line(ix, iy, inx, iny)

                # anti-aliased line
                rr, cc, val = line_aa(ix, iy, inx, iny)

                img[rr, cc] = val

        # print(np.max(img))

        return img

    # def plot_graph(self, name=''):
    #
    #     fig, ax = plt.subplots()
    #
    #     for n in self.nodes:
    #         x = n.data['location'][0]
    #         y = n.data['location'][1]
    #
    #         # Draw a dot at the node location
    #         ax.scatter(x, y)
    #
    #         # Go through the neighbor indices of each neighbor
    #         for ni in n.get_neighbors():
    #             # Draw a line between the current node and the neighbor node
    #             nx = self.nodes[ni].data['location'][0]
    #             ny = self.nodes[ni].data['location'][1]
    #             ax.plot([x, nx], [y, ny])
    #
    #     if name:
    #         fig.suptitle(name)
    #
    #     return fig


class Node(object):

    def __init__(self, index, data=None):

        self.index = index

        # key value pairs of neighbors
        # keys are the unique index of the neighbor, value is the distance
        self.neighbors = {}

        self.data = data

    def add_neighbor(self, neighbor_index, distance):

        self.neighbors[neighbor_index] = distance

    def get_neighbors(self):

        return list(self.neighbors.keys())


if __name__ == '__main__':
    # various tests

    # Simple linear chain of nodes
    n_nodes = 10
    straight_line_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        if i + 1 < n_nodes:
            straight_line_matrix[i, i + 1] = 1
        if i - 1 > 0:
            straight_line_matrix[i, i - 1] = 1

    graph = Graph(matrix=straight_line_matrix)

    path = graph.search_graph(3, 7)
    print(path)

    separated = np.zeros((n_nodes, n_nodes))
    separated[:3, :3] = 1
    separated[-3:, -3:] = 1

    graph = Graph(matrix=separated)

    path = graph.search_graph(1, 8)
    print(path)
    path = graph.search_graph(0, 2)
    print(path)




