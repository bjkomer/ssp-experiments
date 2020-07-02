import numpy as np
import nengo
import nengo_spa as spa
from spatial_semantic_pointers.utils import get_heatmap_vectors, get_fixed_dim_sub_toriod_axes, make_good_unitary, \
    encode_point, ssp_to_loc_v
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from spatial_semantic_pointers.plots import SpatialHeatmap


dim=512
dim=64
n_items=16
seed=0
limit=5
res=128
thresh=0.5
neurons_per_dim=25
time_per_item=1.0
max_items=100


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
    for j in range(int(n_items / (4 ** (n_levels - i - 1)))):
        vocab.populate('LevelFillerID{}_{}.unitary()'.format(i, j))
        # filler_id_keys.append('LevelFillerID{}_{}'.format(i, j))
        # filler_keys.append('LevelFiller{}_{}'.format(i, j))
        # mapping['LevelFillerID{}_{}'.format(i, j)] = 'LevelFiller{}_{}'.format(i, j)

# Second last level with item*location pairs
for i in range(int(n_items / 4)):
    id_str = []
    for k in range(n_levels - 1):
        id_str.append('LevelSlot{} * LevelFillerID{}_{}'.format(k, k, int(i * 4 / (4 ** (n_levels - k - 1)))))

    data_str = []
    for j in range(4):
        ind = i * 4 + j
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

# save time for very large numbers of items
n_exp_items = min(n_items, max_items)
estims = np.zeros((n_exp_items, dim,))
sims = np.zeros((n_exp_items,))

# filler_id_vocab = vocab.create_subset(keys=filler_id_keys)
filler_vocab = vocab.create_subset(keys=filler_keys + filler_id_keys)

# print(filler_keys)
# print(len(filler_keys))

model = nengo.Network(seed=seed)
# model.config.configures(spa.State)
# model.config[spa.State].neuron_type = nengo.Direct()
# model.config[nengo.Ensemble].neuron_type = nengo.Direct()
direct_config = nengo.Config(nengo.Ensemble)
direct_config[nengo.Ensemble].neuron_type = nengo.Direct()
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
        lambda t: vocab['LevelSlot{}'.format(n_levels - 1)].v,
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

    # model.cconv_noisy_level_filler = nengo.networks.CircularConvolution(
    #     n_neurons=neurons_per_dim * 2, dimensions=dim, invert_b=True
    # )
    model.cconv_noisy_level_filler = spa.networks.CircularConvolution(
        n_neurons=neurons_per_dim * 2, dimensions=dim, invert_b=True
    )

    nengo.Connection(item_input_node, model.cconv_noisy_level_filler.input_a)
    nengo.Connection(level_slot_input_node, model.cconv_noisy_level_filler.input_a)

    # Note: this is set up as heteroassociative between ID and the content (should clean up as well)
    model.noisy_level_filler_id_cleanup = spa.ThresholdingAssocMem(
        threshold=0.7,
        # input_vocab=filler_id_vocab,
        input_vocab=filler_vocab,
        # mapping=vocab.keys(),
        mapping=mapping,
        function=lambda x: x > 0.
    )

    nengo.Connection(model.cconv_noisy_level_filler.output, model.noisy_level_filler_id_cleanup.input)

    # model.cconv_location = nengo.networks.CircularConvolution(
    #     n_neurons=neurons_per_dim * 2, dimensions=dim, invert_b=True
    # )
    model.cconv_location = spa.networks.CircularConvolution(
        n_neurons=neurons_per_dim * 2, dimensions=dim, invert_b=True
    )

    nengo.Connection(model.noisy_level_filler_id_cleanup.output, model.cconv_location.input_a)
    nengo.Connection(item_id_input_node, model.cconv_location.input_b)

    out_node = nengo.Node(size_in=dim, size_out=0)

    nengo.Connection(model.cconv_location.output, out_node)

    heatmap_node = nengo.Node(
        SpatialHeatmap(hmv, xs, ys, cmap='plasma', vmin=None, vmax=None),
        size_in=dim, size_out=0,
    )

    nengo.Connection(model.cconv_location.output, heatmap_node)

    with direct_config:

        model.item_id_reader = spa.State(
            vocab,
            # neuron_type=nengo.Direct()
        )
        nengo.Connection(item_id_input_node, model.item_id_reader.input)

        model.item_reader = spa.State(
            vocab,
        )
        nengo.Connection(item_input_node, model.item_reader.input)

        model.level_slot_reader = spa.State(
            vocab,
            # neuron_type=nengo.Direct()
        )
        nengo.Connection(level_slot_input_node, model.level_slot_reader.input)

        model.out_loc_reader = spa.State(
            vocab,
            # neuron_type=nengo.Direct()
        )
        nengo.Connection(model.cconv_location.output, model.out_loc_reader.input)

        model.cleanup_input = spa.State(
            vocab,
            # neuron_type=nengo.Direct()
        )
        nengo.Connection(model.noisy_level_filler_id_cleanup.input, model.cleanup_input.input)



    # model.cleanup_output = spa.State(
    #     vocab,
    #     # neuron_type=nengo.Direct()
    # )
    # nengo.Connection(model.noisy_level_filler_id_cleanup.output, model.cleanup_output.input)
