import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, ssp_to_loc, get_heatmap_vectors
from spatial_semantic_pointers.plots import plot_heatmap
import nengo.spa as spa
from graphs import Graph, Node
from algorithms import EllipticExpansion
import matplotlib.pyplot as plt

import argparse

from tensorboardX import SummaryWriter


def main():

    parser = argparse.ArgumentParser('Traverse many graphs with and SSP algorithm and report metrics')

    parser.add_argument('--n-samples', type=int, default=5, help='Number of different graphs to test')
    parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
    parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
    parser.add_argument('--res', type=int, default=128, help='Resolution of the linspaces used')
    parser.add_argument('--diameter-increment', type=float, default=1.0,
                        help='How much to expand ellipse diameter by on each step')

    args = parser.parse_args()


    # Metrics
    # set to 1 if the found path is the shortest
    shortest_path = np.zeros((args.n_samples))

    # set to 1 if the path found is valid (only uses connections that exist)
    valid_path = np.zeros((args.n_samples))

    # set to 1 if any path is found in the time allotted
    found_path = np.zeros((args.n_samples))

    np.random.seed(args.seed)

    xs = np.linspace(0, 10, args.res)
    ys = np.linspace(0, 10, args.res)

    x_axis_sp = make_good_unitary(args.dim)
    y_axis_sp = make_good_unitary(args.dim)

    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

    # TEMP: putting this outside the loop for debugging
    graph_params = generate_graph(dim=args.dim, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

    graph_params2 = graph_params.copy()
    x_axis_sp2 = x_axis_sp.copy()
    y_axis_sp2 = y_axis_sp.copy()
    xs2 = xs.copy()
    ys2 = ys.copy()
    heatmap_vectors2 = heatmap_vectors.copy()

    for n in range(args.n_samples):

        print("Sample {} of {}".format(n+1, args.n_samples))

        # graph_params = generate_graph(dim=args.dim, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

        elliptic_expansion = EllipticExpansion(
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            xs=xs,
            ys=ys,
            heatmap_vectors=heatmap_vectors,
            diameter_increment=args.diameter_increment,
            debug_mode=True,
            **graph_params
        )

        path = elliptic_expansion.find_path(
            max_steps=10,#15,#20,
            display=False,
            graph=graph_params['graph'],
            xs=xs,
            ys=ys,
            heatmap_vectors=heatmap_vectors
        )

        optimal_path = graph_params['graph'].search_graph(
            start_node=graph_params['start_landmark_id'],
            end_node=graph_params['end_landmark_id'],
        )

        print("found path is: {}".format(path))
        print("optimal path is: {}".format(optimal_path))

        if path is not None:
            found_path[n] = 1

            if graph_params['graph'].is_valid_path(path):
                valid_path[n] = 1
                print("path is valid")
            else:
                print("path is invalid")

            if path == optimal_path:
                shortest_path[n] = 1
                print("path is optimal")
            else:
                print("path is not optimal")

    elliptic_expansion2 = EllipticExpansion(
        x_axis_sp=x_axis_sp2,
        y_axis_sp=y_axis_sp2,
        xs=xs2,
        ys=ys2,
        heatmap_vectors=heatmap_vectors2,
        diameter_increment=args.diameter_increment,
        debug_mode=True,
        **graph_params2
    )

    path2 = elliptic_expansion2.find_path(
        max_steps=10,  # 15,#20,
        display=False,
        graph=graph_params2['graph'],
        xs=xs2,
        ys=ys2,
        heatmap_vectors=heatmap_vectors
    )

    optimal_path2 = graph_params['graph'].search_graph(
        start_node=graph_params['start_landmark_id'],
        end_node=graph_params['end_landmark_id'],
    )

    print("found path is: {}".format(path2))
    print("optimal path is: {}".format(optimal_path2))

    if path2 is not None:
        found_path[n] = 1

        if graph_params['graph'].is_valid_path(path2):
            valid_path[n] = 1
            print("path is valid")
        else:
            print("path is invalid")

        if path2 == optimal_path2:
            shortest_path[n] = 1
            print("path is optimal")
        else:
            print("path is not optimal")

    print("Found path: {}".format(found_path.mean()))
    print("Valid path: {}".format(valid_path.mean()))
    print("Shortest path: {}".format(shortest_path.mean()))


def generate_graph(dim, x_axis_sp, y_axis_sp, normalize=True):
    # TODO: make different graphs and different start/end nodes each time instead of the same one

    # Map
    map_sp = spa.SemanticPointer(data=np.zeros((dim,)))
    # version of the map with landmark IDs bound to each location
    landmark_map_sp = spa.SemanticPointer(data=np.zeros((dim,)))

    # Connectivity
    # contains each connection egocentrically
    con_ego_sp = spa.SemanticPointer(data=np.zeros((dim,)))
    # contains each connection allocentrically
    con_allo_sp = spa.SemanticPointer(data=np.zeros((dim,)))

    # Agent Location
    agent_sp = spa.SemanticPointer(data=np.zeros((dim,)))

    # True values for individual node connections, for debugging
    true_allo_con_sps = list()

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
    vocab = spa.Vocabulary(dim, max_similarity=0.01)
    for i, loc in enumerate(node_locs):
        nodes.append(Node(index=i, data={'location': loc}))

        map_sp += encode_point(loc[0], loc[1], x_axis_sp, y_axis_sp)

        # Note: the landmark IDs don't have to be 'good' unitaries
        # landmark_ids.append(make_good_unitary(dim))
        # landmark_ids.append(spa.SemanticPointer(dim))

        # sp = spa.SemanticPointer(dim)
        # sp.make_unitary()

        sp = vocab.parse("Landmark{}".format(i))
        landmark_ids.append(sp)

        landmark_map_sp += landmark_ids[i] * encode_point(loc[0], loc[1], x_axis_sp, y_axis_sp)

        vocab_vectors[i, :] = landmark_ids[i].v

    if normalize:
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
        links_ego_sp = spa.SemanticPointer(data=np.zeros((dim,)))
        links_allo_sp = spa.SemanticPointer(data=np.zeros((dim,)))
        for j in connectivity_list[i]:
            vec_diff = node_locs[j] - node_locs[i]
            node.add_neighbor(
                neighbor_index=j,
                distance=np.linalg.norm(vec_diff)
            )

            # links_sp += encode_point(vec_diff[0], vec_diff[1], x_axis_sp, y_axis_sp)
            links_ego_sp += landmark_ids[j] * encode_point(vec_diff[0], vec_diff[1], x_axis_sp, y_axis_sp)
            links_allo_sp += landmark_ids[j] * encode_point(node_locs[j][0], node_locs[j][1], x_axis_sp, y_axis_sp)

        if normalize:
            links_ego_sp.normalize()
            links_allo_sp.normalize()
        con_ego_sp += landmark_ids[i] * links_ego_sp
        con_allo_sp += landmark_ids[i] * links_allo_sp

        true_allo_con_sps.append(links_allo_sp)

    if normalize:
        con_ego_sp.normalize()
        con_allo_sp.normalize()

    graph.nodes = nodes
    graph.n_nodes = 7

    return {
        'graph': graph,
        'start_landmark_id': 0,
        'end_landmark_id': 6,
        'landmark_map_sp': landmark_map_sp,
        'con_ego_sp': con_ego_sp,
        'con_allo_sp': con_allo_sp,
        'landmark_vectors': vocab_vectors,
        # params for debugging
        'true_allo_con_sps': true_allo_con_sps,
        'connectivity_list': connectivity_list,
    }


if __name__ == '__main__':
    main()
