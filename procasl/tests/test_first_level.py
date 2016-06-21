import numpy as np
from numpy.testing import (
    assert_almost_equal, assert_equal, assert_array_equal, assert_warns)

from procasl.first_level import (
    compute_perfusion_regressors)


def test_make_regressor_1():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frame_times = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_perfusion_regressors([condition], ['cond'],
                                                  hrf_model,
                                                  frame_times)
    assert_equal(len(reg_names), 2)
    assert_equal(reg_names[0], 'perfusion_baseline')
    assert_equal(reg_names[1], 'perfusion_cond')
    assert_almost_equal(np.sum(reg), -3, 1)


def test_make_regressor_2():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frame_times = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_perfusion_regressors([condition], ['cond'],
                                                  hrf_model,
                                                  frame_times)
    assert_equal(len(reg_names), 2)
    assert_equal(reg_names[0], 'perfusion_baseline')
    assert_equal(reg_names[1], 'perfusion_cond')
    assert_almost_equal(np.sum(reg) * 16, -1.5, 1)


def test_make_regressor_3():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frame_times = np.linspace(0, 138, 70)
    hrf_model = 'fir'
    reg, reg_names = compute_perfusion_regressors([condition], ['cond'],
                                                  hrf_model,
                                                  frame_times,
                                                  fir_delays=np.arange(4))
    assert_equal(len(reg_names), 5)
    assert_array_equal(np.unique(reg), np.array([-.5, 0, .5]))
    assert_array_equal(np.sum(reg, 1), np.array([ 0., -1.5, 1.5, -1.5, 1.5]))
