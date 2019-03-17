import argparse
from arguments import add_parameters

parser = argparse.ArgumentParser('Run 2D supervised path integration experiment')

parser = add_parameters(parser)

args = parser.parse_args()

