from spatial_semantic_pointers.utils import encode_point
import numpy as np

# 3 directions 120 degrees apart
vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]


def to_ssp(v, X, Y):

    return encode_point(v[0], v[1], X, Y).v


def to_bound_ssp(v, item, X, Y):

    return (item * encode_point(v[0], v[1], X, Y)).v


def to_hex_region_ssp(v, X, Y, spacing=4):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for i in range(3):
        ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v

    return ret


def to_band_region_ssp(v, angle, X, Y):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for dx in np.linspace(20./63., 20, 64):
        ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v

    return ret
