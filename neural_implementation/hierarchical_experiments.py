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


def experiment_direct(dim=512, n_hierarchy=3, n_items=16, seed=0, limit=5, res=128, thresh=0.5,
                      neural=False, neurons_per_dim=25, time_per_item=1.0, max_items=100):
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
        sims = np.zeros((n_items,))
        if neural:
            # save time for very large numbers of items
            n_exp_items = min(n_items, max_items)
            estims = np.zeros((n_exp_items, dim,))
            sims = np.zeros((n_exp_items,))

            model = nengo.Network(seed=seed)
            with model:
                input_node = nengo.Node(
                    lambda t: item_vecs[int(np.floor(t)) % n_items, :],
                    size_in=0, size_out=dim
                )
                mem_node = nengo.Node(
                    mem,
                    size_in=0, size_out=dim
                )

                cconv = nengo.networks.CircularConvolution(n_neurons=neurons_per_dim, dimensions=dim, invert_b=True)

                nengo.Connection(mem_node, cconv.input_a)
                nengo.Connection(input_node, cconv.input_b)

                out_node = nengo.Node(size_in=dim, size_out=0)

                nengo.Connection(cconv.output, out_node)

                p_out = nengo.Probe(out_node, synapse=0.01)

            sim = nengo.Simulator(model)
            sim.run(n_exp_items * time_per_item)

            output_data = sim.data[p_out]
            timesteps_per_item = int(time_per_item / 0.001)

            # timestep offset to cancel transients
            offset = 100
            for i in range(n_exp_items):
                estims[i, :] = output_data[i * timesteps_per_item + offset:(i + 1) * timesteps_per_item, :].mean(axis=0)
                sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

            pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

            errors = np.linalg.norm(pred_locs - locations[:n_exp_items, :], axis=1)

            accuracy = len(np.where(errors < thresh)[0]) / n_items

            rmse = np.sqrt(np.mean(errors ** 2))

            sim = np.mean(sims)
        else:
            # retrieve items
            for i in range(n_items):
                estims[i, :] = (mem_sp *~ spa.SemanticPointer(data=item_vecs[i, :])).v

                sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

            pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

            errors = np.linalg.norm(pred_locs - locations, axis=1)

            accuracy = len(np.where(errors < thresh)[0]) / n_items

            rmse = np.sqrt(np.mean(errors**2))

            sim = np.mean(sims)

    elif n_hierarchy == 2:
        # TODO: generate vocab and input sequences

        n_ids = int(np.sqrt(n_items))
        f_n_ids = np.sqrt(n_items)

        id_vecs = rng.normal(size=(n_ids, dim))
        for i in range(n_ids):
            id_vecs[i, :] = id_vecs[i, :] / np.linalg.norm(id_vecs[i, :])

        # items to be included in each ID vec
        item_sums = np.zeros((n_ids, dim))
        item_loc_sums = np.zeros((n_ids, dim))
        for i in range(n_items):
            id_ind = min(i // n_ids, n_ids - 1)
            # id_ind = min(int(i / f_n_ids), n_ids - 1)
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
        sims = np.zeros((n_items,))

        # retrieve items
        for i in range(n_items):
            # noisy ID for the map with this item
            estim_id = (mem_sp *~ spa.SemanticPointer(data=item_vecs[i, :])).v

            # get closest clean match
            id_sims = np.zeros((n_ids,))
            for j in range(n_ids):
                id_sims[j] = np.dot(estim_id, id_vecs[j, :])

            best_ind = np.argmax(id_sims)

            # clean_id = id_vecs[best_ind, :]

            # item_loc_sums comes from the associative mapping from clean_id

            estims[i, :] = (spa.SemanticPointer(data=item_loc_sums[best_ind, :]) *~ spa.SemanticPointer(data=item_vecs[i, :])).v

            sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

        pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

        errors = np.linalg.norm(pred_locs - locations, axis=1)

        accuracy = len(np.where(errors < thresh)[0]) / n_items

        rmse = np.sqrt(np.mean(errors**2))

        sim = np.mean(sims)

    elif n_hierarchy == 3:
        # n_ids = int(np.cbrt(n_items))
        f_n_ids = np.cbrt(n_items)
        n_ids = int(np.ceil(np.cbrt(n_items)))
        n_ids_inner = int(np.ceil(np.sqrt(n_items / n_ids)))
        # f_n_ids = np.cbrt(n_items)

        id_outer_vecs = rng.normal(size=(n_ids, dim))
        id_inner_vecs = rng.normal(size=(n_ids_inner, dim))
        for i in range(n_ids):
            id_outer_vecs[i, :] = id_outer_vecs[i, :] / np.linalg.norm(id_outer_vecs[i, :])
            # for j in range(n_ids):
            #     id_inner_vecs[i*n_ids+j, :] = id_inner_vecs[i*n_ids+j, :] / np.linalg.norm(id_inner_vecs[i*n_ids+j, :])
        for i in range(n_ids_inner):
            id_inner_vecs[i, :] = id_inner_vecs[i, :] / np.linalg.norm(id_inner_vecs[i, :])


        # items to be included in each ID vec
        item_outer_sums = np.zeros((n_ids, dim))
        # item_inner_sums = np.zeros((n_ids*n_ids, dim))
        item_inner_sums = np.zeros((n_ids_inner, dim))
        item_loc_outer_sums = np.zeros((n_ids, dim))
        # item_loc_inner_sums = np.zeros((n_ids*n_ids, dim))
        item_loc_inner_sums = np.zeros((n_ids_inner, dim))
        for i in range(n_items):
            # id_outer_ind = min(i // (n_ids * n_ids), n_ids - 1)
            # id_inner_ind = min(i // n_ids, n_ids * n_ids - 1)

            # id_outer_ind = min(int(i / (f_n_ids * f_n_ids)), n_ids - 1)
            # id_inner_ind = min(int(i / f_n_ids), n_ids * n_ids - 1)

            id_outer_ind = min(int(i / (f_n_ids * f_n_ids)), n_ids - 1)
            id_inner_ind = min(int(i / f_n_ids), n_ids_inner - 1)

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

            # for j in range(n_ids):
            #     # normalize previous memories
            #     item_inner_sums[i*n_ids+j, :] = item_inner_sums[i*n_ids+j, :] / np.linalg.norm(item_inner_sums[i*n_ids+j, :])
            #     item_loc_inner_sums[i*n_ids+j, :] = item_loc_inner_sums[i*n_ids+j, :] / np.linalg.norm(item_loc_inner_sums[i*n_ids+j, :])
            #
            #     mem_inner[i, :] += (
            #             spa.SemanticPointer(data=id_inner_vecs[i*n_ids+j, :]) * spa.SemanticPointer(data=item_inner_sums[i*n_ids+j, :])
            #     ).v

        for j in range(n_ids_inner):
            # normalize previous memories
            item_inner_sums[j, :] = item_inner_sums[j, :] / np.linalg.norm(item_inner_sums[j, :])
            item_loc_inner_sums[j, :] = item_loc_inner_sums[j, :] / np.linalg.norm(item_loc_inner_sums[j, :])

            i = min(int(j / n_ids), n_ids - 1)

            mem_inner[i, :] += (
                    spa.SemanticPointer(data=id_inner_vecs[j, :]) * spa.SemanticPointer(data=item_inner_sums[j, :])
            ).v

            mem_inner[i, :] /= np.linalg.norm(mem_inner[i, :])
        mem_outer /= np.linalg.norm(mem_outer)

        mem_outer_sp = spa.SemanticPointer(data=mem_outer)

        estims = np.zeros((n_items, dim,))
        sims = np.zeros((n_items,))

        if neural:
            # time for each item, in seconds
            time_per_item = 1.0
            model = nengo.Network(seed=seed)
            with model:
                inp_node = nengo.Node('?', size_in=0, size_out=dim)

                estim_outer_id = nengo.Ensemble(dimension=dim, n_neurons=dim*neurons_per_dim)

                out_node = nengo.Node(size_in=dim, size_out=0)

                p_out = nengo.Probe(out_node, synapse=0.01)

            sim = nengo.Simulator(model)
            sim.run(n_items*time_per_item)
        else:
            # non-neural version

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
                id_sims = np.zeros((n_ids_inner))
                for j in range(n_ids_inner):
                    id_sims[j] = np.dot(estim_inner_id, id_inner_vecs[j, :])

                best_ind = np.argmax(id_sims)

                # item_loc_sums comes from the associative mapping from clean_id

                estims[i, :] = (
                        spa.SemanticPointer(data=item_loc_inner_sums[best_ind, :]) * ~ spa.SemanticPointer(data=item_vecs[i, :])
                ).v

                sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

        pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

        errors = np.linalg.norm(pred_locs - locations, axis=1)

        accuracy = len(np.where(errors < thresh)[0]) / n_items

        rmse = np.sqrt(np.mean(errors ** 2))

        sim = np.mean(sims)
    else:
        # 4 split hierarchy

        vocab = spa.Vocabulary(dimensions=dim, pointer_gen=np.random.RandomState(seed=seed))
        filler_id_keys = []
        filler_keys = []
        mapping = {}

        items_left = n_items
        n_levels = 0
        while items_left > 1:
            n_levels += 1
            items_left /= 4

        print(n_levels)

        # level_id_str = '; '.join(['LevelID{}'.format(li) for li in range(n_levels)])
        # vocab.populate(level_id_str)
        #
        # level_ids = []

        # Location Values, labelled SSP
        for i in range(n_items):
            # vocab.populate('Item{}'.format(i))
            vocab.add('Loc{}'.format(i), encode_point(locations[i, 0], locations[i, 1], X, Y).v)


        # level IDs, e.g. CITY, PROVINCE, COUNTRY
        for i in range(n_levels):
            vocab.populate('LevelSlot{}.unitary()'.format(i))
            # sp = spa.SemanticPointer()

        # Item IDs, e.g. Waterloo_ID
        for i in range(n_items):
            vocab.populate('ItemID{}.unitary()'.format(i))



        # level labels (fillers for level ID slots), e.g. Waterloo_ID, Ontario_ID, Canada_ID
        for i in range(n_levels):
            for j in range(int(n_items / (4**(n_levels - i - 1)))):
                vocab.populate('LevelFillerID{}_{}.unitary()'.format(i, j))
                # filler_id_keys.append('LevelFillerID{}_{}'.format(i, j))
                # filler_keys.append('LevelFiller{}_{}'.format(i, j))
                # mapping['LevelFillerID{}_{}'.format(i, j)] = 'LevelFiller{}_{}'.format(i, j)

        # Second last level with item*location pairs
        for i in range(int(n_items / 4)):
            id_str = []
            for k in range(n_levels - 1):
                id_str.append('LevelSlot{} * LevelFillerID{}_{}'.format(k, k, int(i*4 / (4**(n_levels - k - 1)))))

            data_str = []
            for j in range(4):
                ind = i*4 + j
                data_str.append('ItemID{}*Loc{}'.format(ind, ind))
                vocab.populate('Item{} = ({}).normalized()'.format(
                    # i, ' + '.join(id_str + ['LevelSlot{} * LevelFillerID{}_{}'.format(n_levels - 2, n_levels - 2, j)])
                    ind, ' + '.join(id_str + ['LevelSlot{} * LevelFillerID{}_{}'.format(n_levels - 1, n_levels - 1, j)])
                ))

            # vocab.populate('LevelFiller{}_{} = {}'.format(n_levels - 1, i, ' + '.join(data_str)))
            vocab.populate('LevelFiller{}_{} = ({}).normalized()'.format(n_levels - 2, i, ' + '.join(data_str)))

            # only appending the ones used
            filler_id_keys.append('LevelFillerID{}_{}'.format(n_levels - 2, i))
            filler_keys.append('LevelFiller{}_{}'.format(n_levels - 2, i))
            mapping['LevelFillerID{}_{}'.format(n_levels - 2, i)] = 'LevelFiller{}_{}'.format(n_levels - 2, i)

        print(sorted(list(vocab.keys())))

        # Given each ItemID, calculate the corresponding Loc
        # Can map from ItemID{X} -> Item{X}
        # Query based on second last levelID to get the appropriate LevelFillerID
        # map from LevelFillerID -> LevelFiller
        # do the query LevelFiller *~ ItemID{X} to get Loc{X}

        possible_level_filler_id_vecs = np.zeros((int(n_items / 4), dim))
        for i in range(int(n_items / 4)):
            possible_level_filler_id_vecs[i] = vocab['LevelFillerID{}_{}'.format(n_levels - 2, i)].v

        estims = np.zeros((n_items, dim,))
        sims = np.zeros((n_items,))

        if neural:
            # save time for very large numbers of items
            n_exp_items = min(n_items, max_items)
            estims = np.zeros((n_exp_items, dim,))
            sims = np.zeros((n_exp_items,))

            filler_id_vocab = vocab.create_subset(keys=filler_id_keys)
            filler_vocab = vocab.create_subset(keys=filler_keys)
            filler_all_vocab = vocab.create_subset(keys=filler_keys + filler_id_keys)

            # print(filler_keys)
            # print(len(filler_keys))

            model = nengo.Network(seed=seed)
            with model:
                # The changing item query. Full expanded item, not just ID
                item_input_node = nengo.Node(
                    lambda t: vocab['Item{}'.format(int(np.floor(t)) % n_items)].v,
                    size_in=0,
                    size_out=dim
                )
                # item_input_node = spa.Transcode(lambda t: 'Item{}'.format(int(np.floor(t))), output_vocab=vocab)

                # The ID for the changing item query
                item_id_input_node = nengo.Node(
                    lambda t: vocab['ItemID{}'.format(int(np.floor(t)) % n_items)].v,
                    size_in=0,
                    size_out=dim
                )
                # item_id_input_node = spa.Transcode(lambda t: 'ItemID{}'.format(int(np.floor(t))), output_vocab=vocab)

                # Fixed memory based on the level slot to access
                level_slot_input_node = nengo.Node(
                    lambda t: vocab['LevelSlot{}'.format(n_levels - 2)].v,
                    size_in=0,
                    size_out=dim
                )
                # level_slot_input_node = spa.Transcode(lambda t: 'LevelSlot{}'.format(n_levels - 1), output_vocab=vocab)

                # noisy_level_filler_id = nengo.Ensemble(dimensions=dim, n_neurons=dim*neurons_per_dim)
                # model.noisy_level_filler_id = spa.State(
                #     # dimensions=dim,
                #     filler_vocab,
                #     # n_neurons=dim * neurons_per_dim
                # )

                model.cconv_noisy_level_filler = nengo.networks.CircularConvolution(
                    n_neurons=neurons_per_dim*2, dimensions=dim, invert_b=True
                )

                nengo.Connection(item_input_node, model.cconv_noisy_level_filler.input_a)
                nengo.Connection(level_slot_input_node, model.cconv_noisy_level_filler.input_b)

                # Note: this is set up as heteroassociative between ID and the content (should clean up as well)
                model.noisy_level_filler_id_cleanup = spa.ThresholdingAssocMem(
                    threshold=0.4,
                    input_vocab=filler_id_vocab,
                    output_vocab=filler_vocab,
                    # mapping=vocab.keys(),
                    mapping=mapping,
                    function=lambda x: x > 0.
                )

                nengo.Connection(model.cconv_noisy_level_filler.output, model.noisy_level_filler_id_cleanup.input)

                model.cconv_location = nengo.networks.CircularConvolution(
                    n_neurons=neurons_per_dim*2, dimensions=dim, invert_b=True
                )

                nengo.Connection(model.noisy_level_filler_id_cleanup.output, model.cconv_location.input_a)
                nengo.Connection(item_id_input_node, model.cconv_location.input_b)

                out_node = nengo.Node(size_in=dim, size_out=0)

                nengo.Connection(model.cconv_location.output, out_node)

                p_out = nengo.Probe(out_node, synapse=0.01)

            sim = nengo.Simulator(model)
            sim.run(n_exp_items*time_per_item)

            output_data = sim.data[p_out]
            timesteps_per_item = int(time_per_item / 0.001)

            # timestep offset to cancel transients
            offset = 100
            for i in range(n_exp_items):
                estims[i, :] = output_data[i*timesteps_per_item + offset:(i+1)*timesteps_per_item, :].mean(axis=0)
                sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

            pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

            errors = np.linalg.norm(pred_locs - locations[:n_exp_items, :], axis=1)

            accuracy = len(np.where(errors < thresh)[0]) / n_items

            rmse = np.sqrt(np.mean(errors ** 2))

            sim = np.mean(sims)
        else:
            # non-neural version

            # retrieve items
            for i in range(n_items):
                noisy_level_filler_id = vocab['Item{}'.format(i)] *~ vocab['LevelSlot{}'.format(n_levels - 2)]
                # cleanup filler id
                n_fillers = int(n_items / 4)
                sim = np.zeros((n_fillers,))
                for j in range(n_fillers):
                    sim[j] = np.dot(noisy_level_filler_id.v, possible_level_filler_id_vecs[j, :])

                filler_id_ind = np.argmax(sim)

                # query the appropriate filler
                loc_estim = vocab['LevelFiller{}_{}'.format(n_levels - 2, filler_id_ind)] *~ vocab['ItemID{}'.format(i)]

                estims[i, :] = loc_estim.v

                sims[i] = np.dot(estims[i, :], encode_point(locations[i, 0], locations[i, 1], X, Y).v)

            pred_locs = ssp_to_loc_v(estims, hmv, xs, ys)

            errors = np.linalg.norm(pred_locs - locations, axis=1)

            accuracy = len(np.where(errors < thresh)[0]) / n_items

            rmse = np.sqrt(np.mean(errors ** 2))

            sim = np.mean(sims)

    return rmse, accuracy, sim


if __name__ == '__main__':

    # rmse, accuracy = experiment_direct(dim=512, n_hierarchy=3, n_items=32, seed=0, limit=5, res=128)
    # print("RMSE: {}".format(rmse))
    # print("Accuracy: {}".format(accuracy))

    fname = 'data_hier.npz'
    fname = 'data_hier_multi.npz'
    fname = 'data_hier_multi_neural.npz'
    # fname = 'data_hier_multi_neural_low.npz'
    # fname = 'data_hier_multi_neural_test.npz'

    # item_numbers = [8, 16, 32, 64, 128, 256, 512, 1024]
    # item_numbers = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    item_numbers = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    item_numbers = [16, 64, 256, 1024]
    item_numbers = [16, 64, 256]
    # item_numbers = [8, 12, 16, 24, 27, 32, 48, 64, 96, 125, 128, 192, 216, 256, 343, 384, 512, 729, 768, 1000, 1024]
    seeds = [0, 1, 2, 3, 4]
    # seeds = [0]
    # item_numbers = [16]

    hierarchies = [1, 2]
    # hierarchies = [4]
    hierarchies = [1, 4]
    item_numbers = [4]

    if os.path.exists(fname):
        data = np.load(fname)
        rmse = data['rmse']
        acc = data['acc']
        sim = data['sim']
    else:

        rmse = np.zeros((len(hierarchies), len(item_numbers), len(seeds)))
        acc = np.zeros((len(hierarchies), len(item_numbers), len(seeds)))
        sim = np.zeros((len(hierarchies), len(item_numbers), len(seeds)))

        # for hi, n_hierarchy in enumerate([1, 2, 3]):
        for hi, n_hierarchy in enumerate(hierarchies):
            for ni, n_items in enumerate(item_numbers):
                for si, seed in enumerate(seeds):
                    print("{} - {} - {} of {} - {} - {}".format(
                        hi+1, ni+1, si+1, len(hierarchies), len(item_numbers), len(seeds))
                    )
                    if n_items == 4 and n_hierarchy == 4:
                        # this case is the same as the non-hierarchy case
                        rmse[hi, ni, si], acc[hi, ni, si], sim[hi, ni, si] = experiment_direct(
                            dim=512, n_hierarchy=1, n_items=n_items, seed=seed, limit=5, res=128,
                            neural=True
                        )
                    else:
                        rmse[hi, ni, si], acc[hi, ni, si], sim[hi, ni, si] = experiment_direct(
                            dim=512, n_hierarchy=n_hierarchy, n_items=n_items, seed=seed, limit=5, res=128,
                            neural=True
                        )

        np.savez(
            fname,
            rmse=rmse,
            acc=acc,
            sim=sim,
        )

    if fname == 'data_hier_multi_neural.npz':
        # combine with the smaller dataset
        data_small = np.load('data_hier_multi_neural_low.npz')
        rmse_small = data_small['rmse']
        acc_small = data_small['acc']
        sim_small = data_small['sim']

        print(rmse.shape)
        print(rmse_small.shape)

        rmse = np.concatenate([rmse_small, rmse], axis=1)
        acc = np.concatenate([acc_small, acc], axis=1)
        sim = np.concatenate([sim_small, sim], axis=1)

        item_numbers = [4, 16, 64, 256]
        seeds = [0, 1, 2, 3, 4]
        hierarchies = [1, 4]


    df = pd.DataFrame()

    for hi, n_hierarchy in enumerate(hierarchies):
        for ni, n_items in enumerate(item_numbers):
            for si, seed in enumerate(seeds):
                if n_hierarchy == 1:
                    hier_str = 'Standard'
                else:
                    hier_str = 'Hierarchical'
                df = df.append(
                    {
                        # 'Hierarchy': '{} Level'.format(n_hierarchy),
                        'Hierarchy': hier_str,
                        'Items': int(n_items),
                        'Seed': int(seed),
                        'RMSE': rmse[hi, ni, si],
                        'Accuracy': acc[hi, ni, si],
                        'Similarity': sim[hi, ni, si],
                    },
                    ignore_index=True
                )

    final_figure = True

    if final_figure:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 3))
        sns.lineplot(data=df, x='Items', y='RMSE', hue='Hierarchy', ax=ax)

        ax.set(xscale='log')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

        # fig, ax = plt.subplots(2, 1, tight_layout=True)
        # sns.lineplot(data=df, x='Items', y='Accuracy', hue='Hierarchy', ax=ax[0])
        # sns.lineplot(data=df, x='Items', y='RMSE', hue='Hierarchy', ax=ax[1])
        #
        # ax[0].set(xscale='log')
        # ax[1].set(xscale='log')
    else:
        print(df)
        fig, ax = plt.subplots(3, 1, tight_layout=True)

        sns.lineplot(data=df, x='Items', y='Similarity', hue='Hierarchy', ax=ax[0])
        sns.lineplot(data=df, x='Items', y='Accuracy', hue='Hierarchy', ax=ax[1])
        sns.lineplot(data=df, x='Items', y='RMSE', hue='Hierarchy', ax=ax[2])

        ax[0].set(xscale='log')
        ax[1].set(xscale='log')
        ax[2].set(xscale='log')

    sns.despine()

    plt.show()

