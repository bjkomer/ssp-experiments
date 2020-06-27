import numpy as np
import nengo
import nengo_spa as spa
from spatial_semantic_pointers.utils import get_heatmap_vectors, get_fixed_dim_sub_toriod_axes, make_good_unitary, \
    encode_point, ssp_to_loc_v
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# def experiment(dim=512, n_hierarchy=3, n_items=16, seed=0):
#     rng = np.random.RandomState(seed=seed)
#
#     if n_hierarchy == 1:  # no hierarchy case
#         pass
#     elif n_hierarchy == 2:
#         # TODO: generate vocab and input sequences
#
#         avg_size = np.sqrt(n_items)
#
#         mem = '?'
#
#         model = nengo.Network(seed=seed)
#         with model:
#             mem_input = nengo.Node(mem, size_in=0, size_out=dim)
#     elif n_hierarchy == 3:
#         avg_size = np.cbrt(n_items)
#     else:
#         raise NotImplementedError


def experiment_direct(dim=512, n_hierarchy=3, n_items=16, seed=0, limit=5, res=128, thresh=0.5):
    rng = np.random.RandomState(seed=seed)

    X, Y = get_fixed_dim_sub_toriod_axes(
        dim=dim,
        n_proj=3,
        scale_ratio=0,
        scale_start_index=0,
        rng=rng,
        eps=0.001,
    )

    xs = np.linspace(-limit, limit, res)
    ys = np.linspace(-limit, limit, res)
    hmv = get_heatmap_vectors(xs, ys, X, Y)

    item_vecs = rng.normal(size=(n_items, dim))
    for i in range(n_items):
        item_vecs[i, :] = item_vecs[i, :] / np.linalg.norm(item_vecs[i, :])

    locations = rng.uniform(low=-limit, high=limit, size=(n_items, 2))

    if n_hierarchy == 1:  # no hierarchy case

        # Encode items into memory
        mem = np.zeros((dim,))
        for i in range(n_items):
            mem += (spa.SemanticPointer(data=item_vecs[i, :]) * encode_point(locations[i, 0], locations[i, 1], X, Y)).v
        mem /= np.linalg.norm(mem)

        mem_sp = spa.SemanticPointer(data=mem)

        # errors = np.zeros((n_items,))
        estims = np.zeros((n_items, dim, ))

        # retrieve items
        for i in range(n_items):
            estims[i, :] = (mem_sp *~ spa.SemanticPointer(data=item_vecs[i, :])).v

        pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

        errors = np.linalg.norm(pred_locs - locations, axis=1)

        accuracy = len(np.where(errors < thresh)[0]) / n_items

        rmse = np.sqrt(np.mean(errors**2))

    elif n_hierarchy == 2:
        # TODO: generate vocab and input sequences

        n_ids = int(np.sqrt(n_items))

        id_vecs = rng.normal(size=(n_ids, dim))
        for i in range(n_ids):
            id_vecs[i, :] = id_vecs[i, :] / np.linalg.norm(id_vecs[i, :])

        # items to be included in each ID vec
        item_sums = np.zeros((n_ids, dim))
        item_loc_sums = np.zeros((n_ids, dim))
        for i in range(n_items):
            id_ind = min(i // n_ids, n_ids - 1)
            item_sums[id_ind, :] += item_vecs[i, :]
            item_loc_sums[id_ind, :] += (
                    spa.SemanticPointer(data=item_vecs[i, :]) * encode_point(locations[i, 0], locations[i, 1], X, Y)
            ).v

        # Encode id_vecs into memory, each id is bound to something that has similarity to all items in the ID's map
        mem = np.zeros((dim,))
        for i in range(n_ids):
            # normalize previous memories
            item_sums[i, :] = item_sums[i, :] / np.linalg.norm(item_sums[i, :])
            item_loc_sums[i, :] = item_loc_sums[i, :] / np.linalg.norm(item_loc_sums[i, :])

            mem += (spa.SemanticPointer(data=id_vecs[i, :]) * spa.SemanticPointer(data=item_sums[i, :])).v
        mem /= np.linalg.norm(mem)

        mem_sp = spa.SemanticPointer(data=mem)

        estims = np.zeros((n_items, dim, ))

        # retrieve items
        for i in range(n_items):
            # noisy ID for the map with this item
            estim_id = (mem_sp *~ spa.SemanticPointer(data=item_vecs[i, :])).v

            # get closest clean match
            id_sims = np.zeros((n_ids))
            for j in range(n_ids):
                id_sims[j] = np.dot(estim_id, id_vecs[j, :])

            best_ind = np.argmax(id_sims)

            # clean_id = id_vecs[best_ind, :]

            # item_loc_sums comes from the associative mapping from clean_id

            estims[i, :] = (spa.SemanticPointer(data=item_loc_sums[best_ind, :]) *~ spa.SemanticPointer(data=item_vecs[i, :])).v

        pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

        errors = np.linalg.norm(pred_locs - locations, axis=1)

        accuracy = len(np.where(errors < thresh)[0]) / n_items

        rmse = np.sqrt(np.mean(errors**2))

    elif n_hierarchy == 3:
        n_ids = int(np.cbrt(n_items))

        id_outer_vecs = rng.normal(size=(n_ids, dim))
        id_inner_vecs = rng.normal(size=(n_ids*n_ids, dim))
        for i in range(n_ids):
            id_outer_vecs[i, :] = id_outer_vecs[i, :] / np.linalg.norm(id_outer_vecs[i, :])
            for j in range(n_ids):
                id_inner_vecs[i*n_ids+j, :] = id_inner_vecs[i*n_ids+j, :] / np.linalg.norm(id_inner_vecs[i*n_ids+j, :])


        # items to be included in each ID vec
        item_outer_sums = np.zeros((n_ids, dim))
        item_inner_sums = np.zeros((n_ids*n_ids, dim))
        item_loc_outer_sums = np.zeros((n_ids, dim))
        item_loc_inner_sums = np.zeros((n_ids*n_ids, dim))
        for i in range(n_items):
            id_outer_ind = min(i // (n_ids * n_ids), n_ids - 1)
            id_inner_ind = min(i // n_ids, n_ids * n_ids - 1)

            item_outer_sums[id_outer_ind, :] += item_vecs[i, :]
            item_inner_sums[id_inner_ind, :] += item_vecs[i, :]

            item_loc_outer_sums[id_outer_ind, :] += (
                    spa.SemanticPointer(data=item_vecs[i, :]) * encode_point(locations[i, 0], locations[i, 1], X, Y)
            ).v
            item_loc_inner_sums[id_inner_ind, :] += (
                    spa.SemanticPointer(data=item_vecs[i, :]) * encode_point(locations[i, 0], locations[i, 1], X, Y)
            ).v

        # Encode id_vecs into memory, each id is bound to something that has similarity to all items in the ID's map
        mem_outer = np.zeros((dim,))
        mem_inner = np.zeros((n_ids, dim,))
        for i in range(n_ids):
            # normalize previous memories
            item_outer_sums[i, :] = item_outer_sums[i, :] / np.linalg.norm(item_outer_sums[i, :])
            item_loc_outer_sums[i, :] = item_loc_outer_sums[i, :] / np.linalg.norm(item_loc_outer_sums[i, :])

            mem_outer += (
                    spa.SemanticPointer(data=id_outer_vecs[i, :]) * spa.SemanticPointer(data=item_outer_sums[i, :])
            ).v

            for j in range(n_ids):
                # normalize previous memories
                item_inner_sums[i*n_ids+j, :] = item_inner_sums[i*n_ids+j, :] / np.linalg.norm(item_inner_sums[i*n_ids+j, :])
                item_loc_inner_sums[i*n_ids+j, :] = item_loc_inner_sums[i*n_ids+j, :] / np.linalg.norm(item_loc_inner_sums[i*n_ids+j, :])

                mem_inner[i, :] += (
                        spa.SemanticPointer(data=id_inner_vecs[i*n_ids+j, :]) * spa.SemanticPointer(data=item_inner_sums[i*n_ids+j, :])
                ).v

            mem_inner[i, :] /= np.linalg.norm(mem_inner[i, :])
        mem_outer /= np.linalg.norm(mem_outer)

        mem_outer_sp = spa.SemanticPointer(data=mem_outer)

        estims = np.zeros((n_items, dim,))

        # retrieve items
        for i in range(n_items):
            # noisy outer ID for the map with this item
            estim_outer_id = (mem_outer_sp * ~ spa.SemanticPointer(data=item_vecs[i, :])).v

            # get closest clean match
            id_sims = np.zeros((n_ids))
            for j in range(n_ids):
                id_sims[j] = np.dot(estim_outer_id, id_outer_vecs[j, :])

            best_ind = np.argmax(id_sims)

            # noisy inner ID for the map with this item
            estim_inner_id = (spa.SemanticPointer(data=mem_inner[best_ind, :]) * ~ spa.SemanticPointer(data=item_vecs[i, :])).v

            # get closest clean match
            id_sims = np.zeros((n_ids*n_ids))
            for j in range(n_ids*n_ids):
                id_sims[j] = np.dot(estim_inner_id, id_inner_vecs[j, :])

            best_ind = np.argmax(id_sims)

            # item_loc_sums comes from the associative mapping from clean_id

            estims[i, :] = (
                    spa.SemanticPointer(data=item_loc_inner_sums[best_ind, :]) * ~ spa.SemanticPointer(data=item_vecs[i, :])
            ).v

        pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

        errors = np.linalg.norm(pred_locs - locations, axis=1)

        accuracy = len(np.where(errors < thresh)[0]) / n_items

        rmse = np.sqrt(np.mean(errors ** 2))
    else:
        raise NotImplementedError

    return rmse, accuracy


if __name__ == '__main__':

    # rmse, accuracy = experiment_direct(dim=512, n_hierarchy=3, n_items=32, seed=0, limit=5, res=128)
    # print("RMSE: {}".format(rmse))
    # print("Accuracy: {}".format(accuracy))

    fname = 'data_hier.npz'

    item_numbers = [8, 16, 32, 64, 128, 256, 512]
    seeds = [0, 1, 2, 3, 4]

    if os.path.exists(fname):
        data = np.load(fname)
        rmse = data['rmse']
        acc = data['acc']
    else:

        rmse = np.zeros((3, len(item_numbers), len(seeds)))
        acc = np.zeros((3, len(item_numbers), len(seeds)))

        for hi, n_hierarchy in enumerate([1, 2, 3]):
            for ni, n_items in enumerate(item_numbers):
                for si, seed in enumerate(seeds):
                    print("{} - {} - {} of {} - {} - {}".format(hi+1, ni+1, si+1, 3, len(item_numbers), len(seeds)))
                    rmse[hi, ni, si], acc[hi, ni, si] = experiment_direct(
                        dim=512, n_hierarchy=n_hierarchy, n_items=n_items, seed=seed, limit=5, res=128
                    )

        np.savez(
            fname,
            rmse=rmse,
            acc=acc,
        )

    df = pd.DataFrame()

    for hi, n_hierarchy in enumerate([1, 2, 3]):
        for ni, n_items in enumerate(item_numbers):
            for si, seed in enumerate(seeds):
                df = df.append(
                    {
                        'Hierarchy': '{} Level'.format(n_hierarchy),
                        'Items': int(n_items),
                        'Seed': int(seed),
                        'RMSE': rmse[hi, ni, si],
                        'Accuracy': acc[hi, ni, si],
                    },
                    ignore_index=True
                )
    print(df)
    fig, ax = plt.subplots(2, 1, tight_layout=True)

    sns.lineplot(data=df, x='Items', y='RMSE', hue='Hierarchy', ax=ax[0])
    sns.lineplot(data=df, x='Items', y='Accuracy', hue='Hierarchy', ax=ax[1])

    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    plt.show()

