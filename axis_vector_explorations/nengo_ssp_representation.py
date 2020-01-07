import nengo
from spatial_semantic_pointers.utils import encode_point, make_good_unitary
from utils import CompleteSpatialSpikePlot, Spiral


model = nengo.Network(seed=13)

dim = 256

neurons_per_dim = 5

n_neurons = dim * neurons_per_dim

limit = 5

X = make_good_unitary(dim)
Y = make_good_unitary(dim)


def pos_to_ssp(x):
    return encode_point(x[0], x[1], X, Y).v


with model:

    # pos = nengo.Node([0, 0])
    pos = nengo.Node(
        Spiral(
            ang_vel=2.9, lin_acc=.025,
            dt=0.001, vel_max=10, fixed_environment=True,
            xlim=(-limit, limit), ylim=(-limit, limit)
        ),
        size_in=0,
        size_out=4,
    )
    index = nengo.Node([0])

    ssp = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=dim,
        # neuron_type=nengo.LIFRate()
    )

    spike_plot = nengo.Node(
        CompleteSpatialSpikePlot(
            grid_size=256,#512,
            n_neurons=n_neurons,
            xlim=(-limit, limit),
            ylim=(-limit, limit),
        ),
        size_in=n_neurons + 3,  # x, y, index, n_neurons
        size_out=0,
    )

    nengo.Connection(pos[[0, 1]], ssp, function=pos_to_ssp)
    nengo.Connection(pos[[0, 1]], spike_plot[[0, 1]], synapse=None)
    nengo.Connection(index, spike_plot[2], synapse=None)
    nengo.Connection(ssp.neurons, spike_plot[3:], synapse=None)

