from warnings import warn
import numpy as np
import scipy.stats as sps

from .utils import multiple_mahalanobis, z_score


DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


def compute_contrast(labels, regression_result, con_val, contrast_type=None):
    """ Compute the specified contrast given an estimated glm

    Parameters
    ----------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model

    results : dict,
        With keys corresponding to the different labels
        values are RegressionResults instances corresponding to the voxels.

    con_val : numpy.ndarray of shape (p) or (q, p)
        Where q = number of contrast vectors and p = number of regressors.

    contrast_type : {None, 't', 'F'}, optional
        Type of the contrast.  If None, then defaults to 't' for 1D
        `con_val` and 'F' for 2D `con_val`

    Returns
    -------
    con : Contrast instance,
        Yields the statistics of the contrast (effects, variance, p-values)
    """
    con_val = np.asarray(con_val)
    dim = 1
    if con_val.ndim > 1:
        dim = con_val.shape[0]

    if contrast_type is None:
        contrast_type = 't' if dim == 1 else 'F'

    acceptable_contrast_types = ['t', 'F']
    if contrast_type not in acceptable_contrast_types:
        raise ValueError(
            '"{0}" is not a known contrast type. Allowed types are {1}'.
            format(contrast_type, acceptable_contrast_types))

    effect_ = np.zeros((dim, labels.size))
    var_ = np.zeros((dim, dim, labels.size))
    if contrast_type == 't':
        for label_ in regression_result:
            label_mask = labels == label_
            resl = regression_result[label_].Tcontrast(con_val)
            effect_[:, label_mask] = resl.effect.T
            var_[:, :, label_mask] = (resl.sd ** 2).T
    else:
        for label_ in regression_result:
            label_mask = labels == label_
            resl = regression_result[label_].Fcontrast(con_val)
            effect_[:, label_mask] = resl.effect
            var_[:, :, label_mask] = resl.covariance

    dof_ = regression_result[label_].df_resid
    return Contrast(effect=effect_, variance=var_, dof=dof_,
                    contrast_type=contrast_type)


def _fixed_effect_contrast(labels, results, con_vals, contrast_type=None):
    """Computes the summary contrast assuming fixed effects.

    Adds the same contrast applied to all labels and results lists.
    """
    contrast = None
    for i, (lab, res, con_val) in enumerate(zip(labels, results, con_vals)):
        if np.all(con_val == 0):
            warn('Contrast for session %d is null' % i)
            continue
        contrast_ = compute_contrast(lab, res, con_val, contrast_type)
        if contrast is None:
            contrast = contrast_
        else:
            contrast = contrast + contrast_
    if contrast is None:
        raise ValueError('all contrasts provided were null contrasts')
    return contrast


class Contrast(object):
    """ The contrast class handles the estimation of statistical contrasts
    on a given model: student (t) or Fisher (F).
    The important feature is that it supports addition,
    thus opening the possibility of fixed-effects models.

    The current implementation is meant to be simple,
    and could be enhanced in the future on the computational side
    (high-dimensional F constrasts may lead to memory breakage).
    """

    def __init__(self, effect, variance, dof=DEF_DOFMAX, contrast_type='t',
                 tiny=DEF_TINY, dofmax=DEF_DOFMAX):
        """
        Parameters
        ----------
        effect : array of shape (contrast_dim, n_voxels)
            the effects related to the contrast

        variance : array of shape (contrast_dim, contrast_dim, n_voxels)
            the associated variance estimate

        dof : scalar
            the degrees of freedom of the resiudals

        contrast_type: {'t', 'F'}
            specification of the contrast type
        """
        if variance.ndim != 3:
            raise ValueError('Variance array should have 3 dimensions')
        if effect.ndim != 2:
            raise ValueError('Effect array should have 2 dimensions')
        if variance.shape[0] != variance.shape[1]:
            raise ValueError('Inconsistent shape for the variance estimate')
        if ((variance.shape[1] != effect.shape[0]) or
            (variance.shape[2] != effect.shape[1])):
            raise ValueError('Effect and variance have inconsistent shape')

        self.effect = effect
        self.variance = variance
        self.dof = float(dof)
        self.dim = effect.shape[0]
        if self.dim > 1 and contrast_type is 't':
            print('Automatically converted multi-dimensional t to F contrast')
            contrast_type = 'F'
        self.contrast_type = contrast_type
        self.stat_ = None
        self.p_value_ = None
        self.baseline = 0
        self.tiny = tiny
        self.dofmax = dofmax

    def effect_size(self):
        """Make access to summary statistics more straightforward when
        computing contrasts"""
        return self.effect[0, :]

    def effect_variance(self):
        """Make access to summary statistics more straightforward when
        computing contrasts"""
        return self.variance[0, 0, :]

    def stat(self, baseline=0.0):
        """ Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline : float, optional
            Baseline value for the test statistic

        Returns
        -------
        stat: 1-d array, shape=(n_voxels,)
            statistical values, one per voxel
        """
        self.baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny))
            if self.contrast_type == 'F':
                stat = stat ** 2
        # Case: F contrast
        elif self.contrast_type == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t inv(v) e
            if self.effect.ndim == 1:
                self.effect = self.effect[np.newaxis]
            if self.variance.ndim == 1:
                self.variance = self.variance[np.newaxis, np.newaxis]
            stat = (multiple_mahalanobis(
                    self.effect - baseline, self.variance) / self.dim)
        # Unknwon stat
        else:
            raise ValueError('Unknown statistic type')
        self.stat_ = stat
        return stat.ravel()

    def p_value(self, baseline=0.0):
        """Return a parametric estimate of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline : float, optional
            baseline value for the test statistic

        Returns
        -------
        p_values : 1-d array, shape=(n_voxels,)
            p-values, one per voxel
        """
        if self.stat_ is None or not self.baseline == baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.contrast_type == 't':
            p_values = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.contrast_type == 'F':
            p_values = sps.f.sf(self.stat_, self.dim, np.minimum(
                                self.dof, self.dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self.p_value_ = p_values
        return p_values

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline: float, optional,
                  Baseline value for the test statistic

        Returns
        -------
        z_score: 1-d array, shape=(n_voxels,)
            statistical values, one per voxel

        """
        if self.p_value_ is None or not self.baseline == baseline:
            self.p_value_ = self.p_value(baseline)

        # Avoid inf values kindly supplied by scipy.
        self.z_score_ = z_score(self.p_value_)
        return self.z_score_

    def __add__(self, other):
        """Addition of selfwith others, Yields an new Contrast instance
        This should be used only on indepndent contrasts"""
        if self.contrast_type != other.contrast_type:
            raise ValueError(
                'The two contrasts do not have consistant type dimensions')
        if self.dim != other.dim:
            raise ValueError(
                'The two contrasts do not have compatible dimensions')
        effect_ = self.effect + other.effect
        variance_ = self.variance + other.variance
        dof_ = self.dof + other.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    def __rmul__(self, scalar):
        """Multiplication of the contrast by a scalar"""
        scalar = float(scalar)
        effect_ = self.effect * scalar
        variance_ = self.variance * scalar ** 2
        dof_ = self.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    __mul__ = __rmul__

    def __div__(self, scalar):
        return self.__rmul__(1 / float(scalar))
