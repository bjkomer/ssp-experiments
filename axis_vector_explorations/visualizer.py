from utils import make_periodic_axes, Visualizer
import nengo


res = 64


model = nengo.Network(seed=13)
with model:

    seed = nengo.Node([13])

    dimensionality = nengo.Node([32])

    resolution = nengo.Node([32])

    spacing = nengo.Node([5])

    limits = nengo.Node([5])

    vis = nengo.Node(
        Visualizer(),
        size_in=5,
        size_out=0
    )

    nengo.Connection(seed, vis[0])
    nengo.Connection(dimensionality, vis[1])
    nengo.Connection(resolution, vis[2])
    nengo.Connection(spacing, vis[3])
    nengo.Connection(limits, vis[4])
