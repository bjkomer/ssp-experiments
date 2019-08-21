# Visualizer using angle spacing

from utils import AngleSpacingVisualizer
import nengo

model = nengo.Network(seed=13)
with model:

    seed = nengo.Node([13])

    dimensionality = nengo.Node([32])

    resolution = nengo.Node([32])

    angle_spacing = nengo.Node([30, 60])

    angle_offset = nengo.Node([0, 0])

    limits = nengo.Node([5])

    vis = nengo.Node(
        AngleSpacingVisualizer(),
        size_in=8,
        size_out=0
    )

    nengo.Connection(seed, vis[0])
    nengo.Connection(dimensionality, vis[1])
    nengo.Connection(resolution, vis[2])
    nengo.Connection(limits, vis[3])
    nengo.Connection(angle_spacing, vis[[4, 5]])
    nengo.Connection(angle_offset, vis[[6, 7]])
