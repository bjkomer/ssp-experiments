import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
from scipy.linalg import circulant
import nengo_spa as spa

dim = 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if False:

    n_samples = 2000
    points = np.zeros((n_samples, dim))
    # points[0, 0] = 1
    points[0, 0] = 0
    points[0, 3] = 1

    # unitary = -make_good_unitary(dim=dim).v
    unitary = make_good_unitary(dim=dim).v
    u_f = np.zeros((dim,),dtype='complex64')
    u_f[0] = 1
    u_f[1] = np.exp(1.j*np.pi/3./10.0)
    u_f[2] = np.exp(1.j*np.pi/2./10.*1)
    u_f[3] = np.conj(u_f[2])
    u_f[4] = np.conj(u_f[1])
    unitary = np.fft.ifft(u_f).real

    assert np.allclose(np.abs(u_f), 1)

    assert np.allclose(np.fft.fft(unitary), u_f)
    assert np.allclose(np.linalg.norm(unitary), 1)

    rot_mat = circulant(unitary)

    print(unitary)
    print(rot_mat)


    print("det rot mat: {}".format(np.linalg.det(rot_mat)))

    for i in range(1, n_samples):
        points[i, :] = points[i-1, :] @ rot_mat

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.1)
elif False:
    # 2D

    # for k, color in enumerate(['blue', 'red']):
    for k, color in enumerate(['blue']):
        res = 64
        points = np.zeros((res*res, dim))

        if k == 0:
            u_phi_a = np.pi / 3. / 10.
            u_phi_b = np.pi / 3. / 10. * 0

            v_phi_a = np.pi / 2. / 10. * 0
            v_phi_b = np.pi / 2. / 10.
        else:
            u_phi_a = np.pi / 3. / 10.* 0
            u_phi_b = np.pi / 4. / 10.

            v_phi_a = np.pi / 3. / 10.
            v_phi_b = np.pi / 4. / 10.

        u_f = np.zeros((dim,),dtype='complex64')
        u_f[0] = 1
        u_f[1] = np.exp(1.j*u_phi_a)
        u_f[2] = np.exp(1.j*u_phi_b)
        u_f[3] = np.conj(u_f[2])
        u_f[4] = np.conj(u_f[1])
        u = np.fft.ifft(u_f).real
        X = spa.SemanticPointer(data=u)

        v_f = np.zeros((dim,),dtype='complex64')
        v_f[0] = 1
        v_f[1] = np.exp(1.j*v_phi_a)
        v_f[2] = np.exp(1.j*v_phi_b)
        v_f[3] = np.conj(v_f[2])
        v_f[4] = np.conj(v_f[1])
        v = np.fft.ifft(v_f).real
        Y = spa.SemanticPointer(data=v)

        xs = np.linspace(0, 30, res)
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                points[i*res+j] = encode_point(x, y, X, Y).v



        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.1)
else:
    # 2D with colour

    res = 32#64
    points = np.zeros((res*res, dim))


    u_phi_a = np.pi / 3. / 10.
    u_phi_b = np.pi / 3. / 10. * 0

    v_phi_a = np.pi / 2. / 10. * 0
    v_phi_b = np.pi / 2. / 10.


    u_f = np.zeros((dim,),dtype='complex64')
    u_f[0] = 1
    u_f[1] = np.exp(1.j*u_phi_a)
    u_f[2] = np.exp(1.j*u_phi_b)
    u_f[3] = np.conj(u_f[2])
    u_f[4] = np.conj(u_f[1])
    u = np.fft.ifft(u_f).real
    X = spa.SemanticPointer(data=u)

    v_f = np.zeros((dim,),dtype='complex64')
    v_f[0] = 1
    v_f[1] = np.exp(1.j*v_phi_a)
    v_f[2] = np.exp(1.j*v_phi_b)
    v_f[3] = np.conj(v_f[2])
    v_f[4] = np.conj(v_f[1])
    v = np.fft.ifft(v_f).real
    Y = spa.SemanticPointer(data=v)

    xs = np.linspace(0, 60, res)
    ys = np.linspace(0, 30, res)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pt = encode_point(x, y, X, Y).v
            ax.scatter(pt[ 0], pt[1], pt[2], color=(i/res, 0, j/res), alpha=0.1)


plt.show()
