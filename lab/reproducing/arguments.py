def add_parameters(parser):
    """
    Add parameters with defaults from Table 1 of supplementary material
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0102-6/MediaObjects/41586_2018_102_MOESM1_ESM.pdf
    """
    parser.add_argument('--duration', type=float, default=15, help='Duration of simulated trajectories (seconds)')
    parser.add_argument('--env-size', type=float, default=2.2, help='Width and height of environment, or diameter for circular environment (meters)')
    parser.add_argument('--perimeter-distance', type=float, default=0.03, help='Perimeter region distance to walls (meters)')
    parser.add_argument('--lin-vel-rayleigh-scale', type=float, default=0.13, help='Forward velocity Rayleigh distribution scale (m/sec)')
    parser.add_argument('--rot-vel-mean', type=float, default=0, help='Rotation velocity Gaussian distribution mean (deg/sec)')
    parser.add_argument('--rot-vel-std', type=float, default=330, help='Rotation velocity Gaussian distribution standard deviation (deg/sec)')
    parser.add_argument('--perimeter-vel-reduction', type=float, default=.25, help='Velocity reduction factor when located in the perimeter')
    parser.add_argument('--perimeter-ang-change', type=float, default=90, help='Change in angle when located in the perimeter (deg)')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation-step time increment (seconds)')
    parser.add_argument('--n-place-cells', type=int, default=256, help='Number of place cells')
    parser.add_argument('--place-cell-std', type=float, default=0.01, help='Place cell standard deviation parameter (meters)')
    parser.add_argument('--n-hd-cells', type=int, default=12, help='Number of target head direction cells')
    parser.add_argument('--hd-concentration-param', type=float, default=20, help='Head direction concentration parameter')
    parser.add_argument('--grad-clip-thresh', type=float, default=1e-5, help='Gradient clipping threshold')
    parser.add_argument('--minibatch-size', type=int, default=10, help='Number of trajectories used in the calculation of a stochastic gradient')
    parser.add_argument('--trajectory-length', type=int, default=100, help='Number of time steps in the trajectories used for the supervised learning task')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Step size multiplier in the RMSProp algorithm')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')
    parser.add_argument('--regularization-param', type=float, default=1e-5, help='Regularisation parameter for linear layer')
    parser.add_argument('--n-param-updates', type=int, default=300000, help='Total number of gradient descent steps taken')
    return parser
