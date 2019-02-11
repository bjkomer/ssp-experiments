# Traversing a graph from a start node to an end node using SSPs
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, ssp_to_loc, get_heatmap_vectors
from spatial_semantic_pointers.plots import plot_heatmap
import nengo.spa as spa
from graphs import Graph, Node
from algorithms import EllipticExpansion
import matplotlib.pyplot as plt

dim = 512

res = 128
xs = np.linspace(0, 10, res)
ys = np.linspace(0, 10, res)

# These will include the space that is the difference between any two nodes
xs_larger = np.linspace(-10, 10, res)
ys_larger = np.linspace(-10, 10, res)

x_axis_sp = make_good_unitary(dim)
y_axis_sp = make_good_unitary(dim)

heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
heatmap_vectors_larger = get_heatmap_vectors(xs_larger, ys_larger, x_axis_sp, y_axis_sp)

# Map
map_sp = spa.SemanticPointer(data=np.zeros((dim,)))
# version of the map with landmark IDs bound to each location
landmark_map_sp = spa.SemanticPointer(data=np.zeros((dim,)))

# Connectivity
con_sp = spa.SemanticPointer(data=np.zeros((dim,)))

# Agent Location
agent_sp = spa.SemanticPointer(data=np.zeros((dim,)))

# Semantic Pointers for each landmark/node
landmark_ids = list()

# Hardcode a specific graph to work with for prototyping
graph = Graph()

# Create 7 nodes to manually add to the graph
node_locs = list()
node_locs.append(np.array([1.0, 1.0]))
node_locs.append(np.array([1.4, 4.7]))
node_locs.append(np.array([3.2, 6.7]))
node_locs.append(np.array([3.8, 1.4]))
node_locs.append(np.array([4.4, 4.2]))
node_locs.append(np.array([6.7, 1.1]))
node_locs.append(np.array([7.1, 5.0]))
nodes = []
# Vocab of landmark IDs
vocab_vectors = np.zeros((len(node_locs), dim))
for i, loc in enumerate(node_locs):
    nodes.append(Node(index=i, data={'location': loc}))

    map_sp += encode_point(loc[0], loc[1], x_axis_sp, y_axis_sp)

    # Note: the landmark IDs don't have to be 'good' unitaries
    landmark_ids.append(make_good_unitary(dim))

    landmark_map_sp += landmark_ids[i] * encode_point(loc[0], loc[1], x_axis_sp, y_axis_sp)

    vocab_vectors[i, :] = landmark_ids[i].v

map_sp.normalize()
landmark_map_sp.normalize()


connectivity_list = [
    [1, 3],
    [0, 2],
    [1, 4],
    [0, 4],
    [2, 3, 6],
    [6],
    [4, 5],
]


for i, node in enumerate(nodes):
    links_sp = spa.SemanticPointer(data=np.zeros((dim,)))
    for j in connectivity_list[i]:
        vec_diff = node_locs[j] - node_locs[i]
        node.add_neighbor(
            neighbor_index=j,
            distance=np.linalg.norm(vec_diff)
        )

        links_sp += encode_point(vec_diff[0], vec_diff[1], x_axis_sp, y_axis_sp)
    links_sp.normalize()
    con_sp += landmark_ids[i]*links_sp
con_sp.normalize()

graph.nodes = nodes
graph.n_nodes = 7


elliptic_expansion = EllipticExpansion(
    start_landmark_id=0,#landmark_ids[0],
    end_landmark_id=6,#landmark_ids[6],
    landmark_map_sp=landmark_map_sp,
    con_sp=con_sp,
    landmark_vectors=vocab_vectors,
    x_axis_sp=x_axis_sp,
    y_axis_sp=y_axis_sp,
    xs=xs,
    ys=ys,
    heatmap_vectors=heatmap_vectors,
)

path = elliptic_expansion.find_path()

print(path)


# def ssp_graph_search(start_landmark_id, end_landmark_id):
#     current_loc_sp = landmark_map_sp *~ start_landmark_id
#     goal_loc_sp = landmark_map_sp *~ end_landmark_id
#
#     # egocentric displacements to nearby landmarks
#     ego_connections_sp = con_sp *~ start_landmark_id
#
#     # allocentric coordinates of nearby landmarks
#     allo_connections_sp = current_loc_sp * ego_connections_sp
#
#     # IDs of nearby landmarks in a single vector
#     connected_landmarks = landmark_map_sp *~ allo_connections_sp



# misc debugging things

# # figure out the landmark IDs of all neighboring nodes
# fig, ax = plt.subplots(1, 7)
#
# for i, node in enumerate(nodes):
#     result = con_sp *~ landmark_ids[i]
#
#     img = plot_heatmap(result.v, heatmap_vectors_larger, ax[i], xs_larger, ys_larger, name='', vmin=-1, vmax=1, cmap='plasma')
# plt.show()

# # check to see how close the decoding of the map is
# for i, loc in enumerate(node_locs):
#     result = landmark_map_sp *~ landmark_ids[i]
#     loc_decode = ssp_to_loc(result, heatmap_vectors, xs, ys)
#
#     print("Node: {}".format(i))
#     print(loc)
#     print(loc_decode)
#     print("")


# # plot the graph to make sure it was generated correctly
# img = graph.graph_image(xs, ys)
# plt.imshow(img)
# plt.show()
