# exploring the effects of rotating the conjugate part of the SSP
import numpy as np
import matplotlib.pyplot as plt


def unitary_conj_rot(dim=5, index=1, phi=np.pi/2., angle=np.pi/3.):

    fv = np.zeros(dim, dtype='complex64')
    fv[:] = 1
    fv[index] = np.exp(1.j*phi)
    fv[-index] = np.conj(fv[index])

    fv[3] *= np.exp(1.j * angle)
    fv[4] *= np.exp(1.j * angle)

    fv[1] *= np.exp(1.j / angle)
    fv[2] *= np.exp(1.j / angle)

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    return v
    # v = v.real
    # assert np.allclose(np.fft.fft(v), fv)
    # assert np.allclose(np.linalg.norm(v), 1)
    # return v


def unitary_rot(dim=5, phi=np.pi/2., angle=np.pi/3.):
    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1] = np.exp(1.j * phi * np.sin(angle))
    fv[2] = np.exp(1.j * phi * np.cos(angle))
    fv[3] = np.conj(fv[2])
    fv[4] = np.conj(fv[1])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def unitary_angle(dim=7, phi_a=np.pi/2., phi_b=0., angle=np.pi/3.):
    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1] = np.exp(1.j * phi_a)
    fv[2] = np.exp(1.j * phi_b)
    fv[3] = np.exp(1.j * angle)
    fv[4] = np.conj(fv[3])
    fv[5] = np.conj(fv[2])
    fv[6] = np.conj(fv[1])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def zeroed_angle(dim=7, phi_a=np.pi/2., phi_b=0., angle=np.pi/3.):
    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1] = np.exp(1.j * phi_a)
    fv[2] = np.exp(1.j * phi_b)
    fv[3] = 0
    fv[4] = 0
    fv[5] = np.conj(fv[2])
    fv[6] = np.conj(fv[1])

    # assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    # assert np.allclose(np.fft.fft(v), fv)
    # assert np.allclose(np.linalg.norm(v), 1)
    return v


if False:
    dim = 5
    # angle = np.pi/3.
    angle = np.pi/12.
    # angle = 0
    # angle = np.pi/2.
    X = unitary_conj_rot(dim=dim, index=1, phi=np.pi/2., angle=angle)
    Y = unitary_conj_rot(dim=dim, index=2, phi=np.pi/2., angle=angle)

    limit = 5
    res = 64
    xs = np.linspace(-limit, limit, res)
    ys = np.linspace(-limit, limit, res)

    hmv = np.zeros((res, res, dim), dtype='complex64')


    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hmv[i, j, :] = np.fft.ifft(np.fft.fft(X)**x * np.fft.fft(Y)**y)


    plt.imshow(hmv.real[:, :, 0])

    plt.show()
elif False:

    n_angles = 3#32
    angles = np.linspace(0, 2*np.pi, n_angles+1)[:-1]
    dim = 5
    X = np.zeros((dim,))
    Y = np.zeros((dim,))
    for angle in angles:
        X += unitary_rot(dim=5, phi=np.pi / 2., angle=angle)
        Y += unitary_rot(dim=5, phi=np.pi / 2., angle=angle + np.pi/2.)

    limit = 5
    res = 64
    xs = np.linspace(-limit, limit, res)
    ys = np.linspace(-limit, limit, res)
    hmv = np.zeros((res, res, dim))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hmv[i, j, :] = np.fft.ifft(np.fft.fft(X) ** x * np.fft.fft(Y) ** y)

    plt.imshow(hmv.real[:, :, 0])

    plt.show()

elif True:

    n_angles = 5  # 32
    angles = np.linspace(0, 2 * np.pi, n_angles + 1)[:-1]
    dim = 7
    default_angle = 0.
    if True:
        X = unitary_angle(dim=dim, phi_a=np.pi/2., phi_b=0., angle=default_angle)
        Y = unitary_angle(dim=dim, phi_a=0., phi_b=np.pi/2., angle=default_angle)
        # X = zeroed_angle(dim=dim, phi_a=np.pi / 2., phi_b=0., angle=default_angle)
        # Y = zeroed_angle(dim=dim, phi_a=0., phi_b=np.pi / 2., angle=default_angle)
    else:
        X = np.zeros((dim,))
        Y = np.zeros((dim,))
        for angle in angles:
            X += unitary_angle(dim=dim, phi_a=np.pi/2., phi_b=0., angle=angle)
            Y += unitary_angle(dim=dim, phi_a=0., phi_b=np.pi/2., angle=angle)
        X /= n_angles
        Y /= n_angles

    limit = 5
    res = 64
    xs = np.linspace(-limit, limit, res)
    ys = np.linspace(-limit, limit, res)
    hmv = np.zeros((res, res, dim))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hmv[i, j, :] = np.fft.ifft(np.fft.fft(X) ** x * np.fft.fft(Y) ** y).real

    new_ang = np.pi/6.
    X_a = unitary_angle(dim=dim, phi_a=np.pi / 2., phi_b=0., angle=new_ang)
    Y_a = unitary_angle(dim=dim, phi_a=0., phi_b=np.pi / 2., angle=new_ang)
    # X_a = zeroed_angle(dim=dim, phi_a=np.pi / 2., phi_b=0., angle=default_angle)
    # Y_a = zeroed_angle(dim=dim, phi_a=0., phi_b=np.pi / 2., angle=default_angle)

    vec = np.fft.ifft(np.fft.fft(X_a) ** x * np.fft.fft(Y_a) ** y).real

    sim = np.tensordot(vec, hmv, axes=([0], [2]))

    plt.imshow(sim)

    plt.show()
