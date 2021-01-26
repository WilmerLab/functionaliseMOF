import numpy as np
from pytest import approx

from mofun import quaternion_from_two_axes


def test_quaternion_from_two_axes__with_antiparallel_z_axes_inverts_z_coord():
    q = quaternion_from_two_axes((0.,0.,2.),(0.,0.,-2.))
    assert q.apply([1., 1., 1.])[2] == approx(-1., 1e-5)

def test_quaternion_from_two_axes__with_almost_antiparallel_z_axes_mostly_inverts_yz_coords():
    q = quaternion_from_two_axes((0.,0.,1.),(0.,0.00001,-0.99999))
    assert np.isclose(q.as_quat(), [-1., 0., 0., 0.], atol=1e-5).all()
    assert np.isclose(q.apply([1., 1., 1.]), [1., -1., -1.]).all()

def test_quaternion_from_two_axes__with_parallel_axes_is_donothing_0001():
    q = quaternion_from_two_axes((0., 0., 2.),(0., 0., 2.))
    assert np.isclose(q.as_quat(), [0., 0., 0., 1.]).all()
    assert (q.apply([1., 1., 1.]) == [1., 1., 1.]).all()

def test_quaternion_from_two_axes__with_almost_parallel_axes_is_almost_donothing_0001():
    q = quaternion_from_two_axes((0., 0., 1.),(0., 0.00001, 0.99999))
    assert np.isclose(q.as_quat(), [0., 0., 0., 1.], atol=1e-5).all()
    assert np.isclose(q.apply([1., 1., 1.]), [1., 1., 1.]).all()