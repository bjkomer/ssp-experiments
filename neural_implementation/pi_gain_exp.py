# learning the path integration gain based on a slowly modified landmark on a circular track
import nengo
import argparse

parser = argparse.ArgumentParser('Path integration gain learning experiment')

parser.add_argument('--dim', default=5)
parser.add_argument('--neurons-per-dim', default=5)

args = parser.parse_args()

# just using a 1D SSP for simplicity, since movement is in a 1D circular space

# constant running speed and fixed radius track (exact position is known based on time)
# landmark velocity slowly changes throughout the experiment
# defined by a frequency of how often it appears

model = nengo.Network(seed=13)
with model:
    pos = nengo.Ensemble(n_neurons=args.neurons_per_dim*args.dim, dimensions=args.dim)
    vel = nengo.Ensemble(n_neurons=100, dimensions=1)

    integ = nengo.Ensemble(n_neurons=args.neurons_per_dim*args.dim+100, dimensions=args.dim+1)

    nengo.Connection(pos, integ[:args.dim], transform=fft)
    nengo.Connection(vel, integ[-1])

    nengo.Connection(integ, pos, function=scaled_ifft)

    vel_input = nengo.Node([1])

    nengo.Connection(vel_input, vel)

