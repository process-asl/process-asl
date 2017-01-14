from procasl import datasets
import numpy as np
from nose import with_setup
from nose.tools import assert_equal, assert_true, assert_not_equal
from numpy.testing import assert_array_equal
from rodmri.data_fetchers import mri

from nilearn.datasets import utils
from nilearn.datasets.tests import test_utils as tst


def setup_mock():
    return tst.setup_mock(utils, mri)


def teardown_mock():
    return tst.teardown_mock(utils, mri)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_kirby():
    # First session, all subjects
    kirby = datasets.fetch_kirby(subjects=range(21), data_dir=tst.tmpdir,
                                 verbose=1)
    assert_equal(len(tst.mock_url_request.urls), 21)
    assert_equal(len(kirby.anat), 21)
    assert_equal(len(kirby.asl), 21)
    assert_equal(len(kirby.m0), 21)
    assert_true(np.all(np.asarray(kirby.session) == 1))

    # Both sessions, 12 subjects
    tst.mock_url_request.reset()
    kirby = mri.fetch_csd(data_dir=tst.tmpdir, sessions=[1, 2],
                        subjects=range(12), verbose=0)
    # Session 1 has already been downloaded
    assert_equal(len(tst.mock_url_request.urls), 24)
    assert_equal(len(kirby.anat), 24)
    assert_equal(len(kirby.asl), 24)
    assert_equal(len(kirby.m0), 24)
    s = np.asarray(kirby.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_not_equal(kirby.description, '')
