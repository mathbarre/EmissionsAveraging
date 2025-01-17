# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: based on code from Hicham Janati <hicham.janati@inria.fr>
# https://github.com/PythonOT/POT/blob/master/ot/unbalanced.py

import warnings
import numpy as np
from numba import njit

def barycenter_unbalanced_stabilized(A, M, reg, reg_m, weights=None, tau=1e3,
                                     numItermax=1000, stopThr=1e-6,
                                     verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
        Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of
        matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
        the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
        The algorithm used for solving the problem is the generalized
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    M : np.ndarray (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m : float
        Marginal relaxation term > 0
    tau : float
        Stabilization threshold for log domain absorption.
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    q : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré,
        G. (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.


    """
    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    fi = reg_m / (reg_m + reg)

    u = np.ones((dim, n_hists)) / dim
    v = np.ones((dim, n_hists)) / dim

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = np.zeros(dim)
    beta = np.zeros(dim)
    q = np.ones(dim) / dim
    for i in range(numItermax):
        qprev = q.copy()
        Kv = K.dot(v)
        f_alpha = np.exp(- alpha / (reg + reg_m))
        f_beta = np.exp(- beta / (reg + reg_m))
        f_alpha = f_alpha[:, None]
        f_beta = f_beta[:, None]
        u = ((A / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = K.T.dot(u)
        q = (Ktu ** (1 - fi)) * f_beta
        q = q.dot(weights) ** (1 / (1 - fi))
        Q = q[:, None]
        v = ((Q / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if (u > tau).any() or (v > tau).any():
            absorbing = True
            print("absorbing")
            alpha = alpha + reg * np.log(np.max(u, 1))
            beta = beta + reg * np.log(np.max(v, 1))
            K = np.exp((alpha[:, None] + beta[None, :] -
                        M) / reg)
            v = np.ones_like(v)
        Kv = K.dot(v)
        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            q = qprev
            break
        if (i % 10 == 0 and not absorbing) or i == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = abs(q - qprev).max() / max(abs(q).max(),
                                             abs(qprev).max(), 1.)
            if log:
                log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
            if err < stopThr:
                break

    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg` or a lower mass `reg_m`." +
                      "Or a larger absorption threshold `tau`.")
    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced_sinkhorn_dev(A, Ms, reg, reg_m, weights=None,
                                       numItermax=1000, stopThr=1e-6,
                                       verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A.

     The function solves the following optimization problem with a

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
    Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    Ms : np.ndarray (dim, dim,n_hist)
        ground metrics matrices for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    q : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """
    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}
    # Ks is (dim,dim,n_hist)
    Ks = np.exp(- Ms / reg)
    # Ks = Ks.reshape((dim*n_hists,dim))
    Kts = np.swapaxes(Ks, axis1=0, axis2=1)
    # Kts = Kts.reshape((dim*n_hists,dim))

    fi = reg_m / (reg_m + reg)

    v = np.ones((dim, n_hists))/dim
    u = np.ones((dim, 1))/dim
    q = np.ones(dim)
    err = 1.

    for i in range(numItermax):
        uprev = u.copy()
        vprev = v.copy()
        qprev = q.copy()
        Kv = np.diagonal(np.tensordot(Ks, v, axes=([1], [0])),
                         axis1=1, axis2=2)
        u = (A / Kv) ** fi
        Ktu = np.diagonal(np.tensordot(Kts, u, axes=([1], [0])),
                          axis1=1, axis2=2)
        q = ((Ktu ** (1 - fi)).dot(weights))
        q = q ** (1 / (1 - fi))
        Q = q[:, None]
        v = (Q / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            # q = qprev
            break
            # compute change in barycenter
        err = abs(q - qprev).max()
        err /= max(abs(q).max(), abs(qprev).max(), 1.)
        if log:
            log['err'].append(err)
        # if barycenter did not change + at least 10 iterations - stop
        if err < stopThr and i > 10:
            break

        if verbose:
            if i % 10 == 0:
                print(
                    '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(i, err))

    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced_stabilized_dev(A, Ms, reg, reg_m, weights=None,
                                         tau=1e8, numItermax=1000,
                                         stopThr=1e-6, verbose=False,
                                         log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
        Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of
        matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
        the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
        The algorithm used for solving the problem is the generalized
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    Ms : np.ndarray (dim, dim,n_hist) or (dim, n_hist)
        ground metric matrices for OT.
    reg : float
        Entropy regularization term > 0
    reg_m : float
        Marginal relaxation term > 0
    tau : float
        Stabilization threshold for log domain absorption.
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    q : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré,
        G. (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.


    """
    if Ms.ndim < 3:
        return barycenter_unbalanced_stabilized(A, Ms, reg, reg_m, weights,
                                                tau, numItermax, stopThr,
                                                verbose, log)
    else:

        dim, n_hists = A.shape
        if weights is None:
            weights = np.ones(n_hists) / n_hists
        else:
            assert(len(weights) == A.shape[1])

        if log:
            log = {'err': []}

        fi = reg_m / (reg_m + reg)

        u = np.ones((dim, n_hists)) / dim
        v = np.ones((dim, n_hists)) / dim

        # print(reg)
        # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
        Ks = np.exp(- Ms / reg)
        # Ks = Ks.reshape((dim*n_hists,dim))
        Kts = np.swapaxes(Ks, axis1=0, axis2=1)

        fi = reg_m / (reg_m + reg)

        cpt = 0
        err = 1.
        alpha = np.zeros(dim)
        beta = np.zeros(dim)
        q = np.ones(dim) / dim
        for i in range(numItermax):
            qprev = q.copy()
            Kv = np.diagonal(np.tensordot(Ks, v, axes=([1], [0])), axis1=1, axis2=2)
            f_alpha = np.exp(- alpha / (reg + reg_m))
            f_beta = np.exp(- beta / (reg + reg_m))
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
            u = ((A / (Kv + 1e-10)) ** fi) * f_alpha
            Ktu = np.diagonal(np.tensordot(Kts, u, axes=([1], [0])),
                              axis1=1, axis2=2)
            q = (Ktu ** (1 - fi)) * f_beta
            q = q.dot(weights) ** (1 / (1 - fi))
            Q = q[:, None]
            v = ((Q / (Ktu + 1e-10)) ** fi) * f_beta
            absorbing = False
            if (u > tau).any() or (v > tau).any():
                absorbing = True
                alpha = alpha + reg * np.log(np.max(u, 1))
                beta = beta + reg * np.log(np.max(v, 1))
                Mab = ((alpha[:, None] + beta[None, :]).reshape((dim*dim, 1)) -
                       Ms.reshape((dim*dim), n_hists)).reshape((dim, dim, n_hists))
                Ks = np.exp(Mab / reg)
                v = np.ones_like(v)
            Kv = np.diagonal(np.tensordot(Ks, v, axes=([1], [0])),
                             axis1=1, axis2=2)
            if (np.any(Ktu == 0.)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn('Numerical errors at iteration %s' % cpt)
                q = qprev
                break
            if (i % 10 == 0 and not absorbing) or i == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                err = abs(q - qprev).max() / max(abs(q).max(),
                                                 abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                if verbose:
                    if i % 50 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))
                if err < stopThr:
                    break

        if err > stopThr:
            warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                          "Try a larger entropy `reg` or a lower mass `reg_m`." +
                          "Or a larger absorption threshold `tau`.")
        if log:
            log['niter'] = i
            log['logu'] = np.log(u + 1e-300)
            log['logv'] = np.log(v + 1e-300)
            return q, log
        else:
            return q


def barycenter_sinkhorn_dev(A, Ms, reg, weights=None,
                            numItermax=1000, stopThr=1e-6,
                            verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A.

     The function solves the following optimization problem with a

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the balanced entropic regularized
    Wasserstein distance (see ot.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    Ms : np.ndarray (dim, dim,n_hist)
        ground metrics matrices for OT.
    reg : float
        Entropy regularization term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) ndarray
        Balanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """

    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}
    # Ks is (dim,dim,n_hist)
    Ks = np.exp(- Ms / reg)
    # Ks = Ks.reshape((dim*n_hists,dim))
    Kts = np.swapaxes(Ks, axis1=0, axis2=1)
    # Kts = Kts.reshape((dim*n_hists,dim))

    v = np.ones((dim, n_hists))/dim
    u = np.ones((dim, 1))/dim
    q = np.ones(dim)
    err = 1.

    for i in range(numItermax):
        uprev = u.copy()
        vprev = v.copy()
        qprev = q.copy()

        # Kv = K.dot(v)
        Kv = np.diagonal(np.tensordot(Ks, v, axes=([1], [0])),
                         axis1=1, axis2=2)

        # Kv = (Ks.dot(v).reshape(dim,n_hists)
        u = A / Kv
        # Ktu = K.T.dot(u)
        Ktu = np.diagonal(np.tensordot(Kts, u, axes=([1], [0])),
                          axis1=1, axis2=2)
        # Ktu = (Kts.dot(u.flatten())).reshape(dim,n_hists)
        q = np.exp(np.dot(np.log(Ktu), weights.T))
        Q = q[:, None]
        v = (Q / Ktu)

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            # q = qprev
            break
            # compute change in barycenter
        err = abs(q - qprev).max()
        err /= max(abs(q).max(), abs(qprev).max(), 1.)
        if log:
            log['err'].append(err)
        # if barycenter did not change + at least 10 iterations - stop
        if err < stopThr and i > 10:
            break

        if verbose:
            if i % 10 == 0:
                print(
                    '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(i, err))

    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_sinkhorn(A, M, reg, weights=None,
                        numItermax=1000, stopThr=1e-6,
                        verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A.

     The function solves the following optimization problem with a

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the balanced entropic regularized
    Wasserstein distance (see ot.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    Ms : np.ndarray (dim, dim,n_hist)
        ground metrics matrices for OT.
    reg : float
        Entropy regularization term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) ndarray
        Balanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """
    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}
    # Ks is (dim,dim,n_hist)
    K = np.exp(- M / reg)

    v = np.ones((dim, n_hists))
    u = np.ones((dim, 1))
    q = np.ones(dim)
    err = 1.

    for i in range(numItermax):
        uprev = u.copy()
        vprev = v.copy()
        qprev = q.copy()

        # Kv = K.dot(v)
        Kv = K.dot(v)

        # Kv = (Ks.dot(v).reshape(dim,n_hists)
        u = A / Kv
        # Ktu = K.T.dot(u)
        Ktu = (K.T).dot(u)
        # Ktu = (Kts.dot(u.flatten())).reshape(dim,n_hists)
        q = np.exp(np.log(Ktu).dot(weights))
        Q = q[:, None]
        v = (Q / Ktu)

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            # q = qprev
            break
            # compute change in barycenter
        err = abs(q - qprev).max()
        err /= max(abs(q).max(), abs(qprev).max(), 1.)
        if log:
            log['err'].append(err)
        # if barycenter did not change + at least 10 iterations - stop
        if err < stopThr and i > 10:
            break

        if verbose:
            if i % 10 == 0:
                print(
                    '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(i, err))

    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


@njit
def logsumexp_stream(X):
    """
    stream log sum exp from
    https://stackoverflow.com/questions/59850770/fastest-python-log-sum-exp-in-a-reduceat
    ----------
    Input
    X : np.ndarray (dimx, dimy, n)
    ----------
    Output
    res : np.ndarray(dimx, dimy)
        res[i,j] = log(Sum_k exp(X[i,j,k]))
    """
    (dimx, dimy, _) = X.shape
    res = np.empty((dimx, dimy), dtype=X.dtype)
    for i in range(dimx):
        for j in range(dimy):
            alpha = -np.Inf
            r = 0.0
            for x in X[i, j, :]:
                if x != -np.Inf:
                    if x <= alpha:
                        r += np.exp(x - alpha)
                    else:
                        r *= np.exp(alpha - x)
                        r += 1.0
                        alpha = x
            res[i, j] = np.log(r) + alpha
    return res


def prod_separable_logspace(Cx, Cy, gamma, v):
    """
    implementation of Algorithm 3 of
    Wasserstein Dictionary Learning: Optimal Transport-based unsupervised
    non-linear dictionary learning
    (Morgan Schmitz, Matthieu Heitz, Nicolas Bonneel,
    Fred Maurice Ngolè Mboula, David Coeurjolly, Marco Cuturi,
    Gabriel Peyré, Jean-Luc Starck)
    ----------
    Input
    Cx : np.ndarray (dimx, dimx)
        x part of separable cost matrix.
    Cy : np.ndarray (dimy, dimy)
        y part of separable cost matrix.
    gamma : float
        regularization parameter.
    v : np.ndarray(dimx,dimy,n_hist)
        input matrix in logspace
    ----------
    Output
    r : np.ndarray(dimx,dimy)
        result of product (exp(-Cx/gamma)X exp(-Cy/gamma))*v in logspace
    """
    n_hist = v.shape[2]
    R = np.zeros(v.shape)
    for i in range(n_hist):
        if Cy.ndim > 2:
            x = -Cy[:, :, i]/gamma + v[:, np.newaxis, :, i]
        else:
            x = -Cy[:, :]/gamma + v[:, np.newaxis, :, i]
        # mx = np.max(x,axis=2)
        # mx[mx==-np.inf]=0
        # A = np.log(np.exp(x-mx[:,:,None]).sum(axis=2))+mx
        A = logsumexp_stream(x)
        if Cx.ndim > 2:
            y = -Cx[:, np.newaxis, :, i]/gamma + A.T
        else:
            y = -Cx[:, np.newaxis, :]/gamma + A.T
        # my = np.max(y,axis=2)
        # my[my==-np.inf]=0
        # R[:,:,i] = np.log(np.exp(y-my[:,:,None]).sum(axis=2))+my
        R[:, :, i] = logsumexp_stream(y)
    return R


# @njit
# def nb_max_axis_0(arr):
#     # max that works with numba
#     n = arr.shape[0]
#     mx = arr[0]
#     for i in range(n):
#         mx = np.maximum(mx,arr[i])
#     return mx

# @njit
# def nb_prod_separable_logspace(Cx,Cy,gamma,v):
    
#     dimx,dimy,n_hist,_ = v.shape
#     R = np.zeros((dimx,dimy,n_hist))
#     for i in range(n_hist):
#         x = np.zeros((dimx,dimy,dimy))
#         for l in range(dimy):
#             x[:,:,l] = -Cy[:,l]/gamma + v[:,l,i,:]
#         # mx = nb_max_axis_0(x)
#         # for xi in range(dimx):
#         #     for yi in range(dimy):
#         #         if mx[xi,yi] == -np.inf :
#         #             mx[xi,yi]=0
#         # A = np.log(np.exp(x-mx).sum(axis=0))+mx
#         A = logsumexp_stream(x)
#         y = np.zeros((dimx,dimy,dimx))
#         for k in range(dimx):
#             y[:,:,k] = -Cx[:,k,:]/gamma + A[k,:]
#         # my = nb_max_axis_0(y)
#         # for xi in range(dimx):
#         #     for yi in range(dimy):
#         #         if my[xi,yi] == -np.inf :
#         #             my[xi,yi]=0
#         # R[:,:,i] = np.log(np.exp(y-my).sum(axis=0))+my
#         R[:,:,i] = logsumexp_stream(y)
#     return R


def barycenter_unbalanced_sinkhorn2D(A, Cx, Cy, reg, reg_m, weights=None,
                                     numItermax=1000, stopThr=1e-6,
                                     verbose=False, log=False, logspace=True,
                                     reg_K=1e-16):
    """
    ----------
    A : np.ndarray (dimx,dimy, n_hists)
        `n_hists` training distributions a_i of dimension dimxdim
    Cx : np.ndarray (dimx, dimx)
        x part of separable cost matrix for OT.
    Cy : np.ndarray (dimy, dimy)
        y part of separable cost matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    logspace : bool, optional
        compuation done in logspace if True
    Returns
    -------
    q : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.
    """
    dimx, dimy, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[2])
    weights = weights/weights.sum()
    if log:
        log = {'err': []}

    # K = np.exp(- M / reg)

    fi = reg_m / (reg_m + reg)

    if logspace:
        v = np.zeros((dimx, dimy, 1))*np.log(1/dimx/dimy)
        u = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        q = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        err = 1.
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            lKv = prod_separable_logspace(Cx, Cy, reg, v)
            # u = (A / Kv) ** fi
            u = fi*(np.log(A)-np.maximum(lKv, np.log(reg_K)))
            # Ktu = K.T.dot(u)
            lKtu = prod_separable_logspace(Cx.T, Cy.T, reg, u)
            # q = ((Ktu ** (1 - fi)).dot(weights))
            mlktu = (1-fi)*np.max(lKtu, axis=2)
            q = (1 / (1 - fi))*((np.log(np.average(np.exp((1-fi)*lKtu
                                                          - mlktu[:, :, None]),
                                                   axis=2, weights=weights)))
                                + mlktu)
            # q = q ** (1 / (1 - fi))
            Q = q[:, :, np.newaxis]
            # v = (Q / Ktu) ** fi
            v = fi*(Q-np.maximum(lKtu, np.log(reg_K)))

            if (np.any(lKtu == -np.inf)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                # q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(np.exp(qprev)*(1-np.exp(q-qprev))).max()
                err /= max(abs(np.exp(q)).max(), abs(np.exp(qprev)).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))

        if log:
            log['niter'] = i
            log['logu'] = ((u + 1e-300))
            log['logv'] = ((v + 1e-300))
            return np.exp(q), log
        else:
            return np.exp(q)
    else:
        v = np.ones((dimx, dimy, 1))/dimx/dimy
        u = np.ones((dimx, dimy))/dimx/dimy
        q = np.ones((dimx, dimy))/dimx/dimy
        err = 1.
        Kx = np.exp(-Cx/reg)
        Ky = np.exp(-Cy/reg)
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            Kv = np.tensordot(np.tensordot(Kx, v, axes=([1], [0])),
                              Ky, axes=([1], [0])).swapaxes(1, 2)
            u = (A / (Kv+reg_K)) ** fi
            Ktu = np.tensordot(np.tensordot(Kx.T, u, axes=([1], [0])),
                               Ky.T, axes=([1], [0])).swapaxes(1, 2)

            q = ((Ktu ** (1 - fi)).dot(weights))

            q = q ** (1 / (1 - fi))
            Q = q[:, :, None]
            v = (Q / (Ktu+reg_K)) ** fi

            if (np.any(Ktu == 0)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(q-qprev).max()
                err /= max(abs(q).max(), abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))
        if log:
            log['niter'] = i
            log['logu'] = (np.log(u + 1e-300))
            log['logv'] = (np.log(v + 1e-300))
            return q, log
        else:
            return q


@njit
def prod_sep(Kx, Ky, v):
    n_hist = v.shape[2]
    R = np.zeros(v.shape)
    for i in range(n_hist):
        R[:, :, i] = np.dot(np.dot(Kx[:, :, i], v[:, :, i]),
                            Ky[:, :, i])
    return R


def barycenter_unbalanced_sinkhorn2D_wind(A, Cxs, Cys, reg, reg_m,
                                          weights=None, numItermax=1000,
                                          stopThr=1e-6, verbose=False,
                                          log=False, logspace=False,
                                          reg_K=1e-16):
    """
    ----------
    A : np.ndarray (dimx,dimy, n_hists)
        `n_hists` training distributions a_i of dimension dimxdim
    Cxs : np.ndarray (dimx, dimx,n_hist)
        x part of separable cost matrix for OT.
    Cys : np.ndarray (dimy, dimy,n_hist)
        y part of separable cost matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    logspace : bool, optional
        compuation done in logspace if True
    Returns
    -------
    q : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.
    """
    dimx, dimy, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[2])

    if log:
        log = {'err': []}

    # K = np.exp(- M / reg)

    fi = reg_m / (reg_m + reg)

    if logspace:
        v = np.zeros((dimx, dimy, n_hists))*np.log(1/dimx/dimy)
        u = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        q = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        err = 1.
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            lKv = prod_separable_logspace(Cxs, Cys, reg, v)
            # u = (A / Kv) ** fi
            u = fi*(np.log(A)-np.maximum(lKv, np.log(reg_K)))
            # Ktu = K.T.dot(u)
            lKtu = prod_separable_logspace(Cxs.swapaxes(0, 1),
                                           Cys.swapaxes(0, 1), reg, u)
            # q = ((Ktu ** (1 - fi)).dot(weights))
            mlktu = (1-fi)*np.max(lKtu, axis=2)
            q = (1 / (1 - fi))*((np.log(np.average(np.exp((1-fi)*lKtu -
                                                          mlktu[:, :, None]),
                                                   axis=2, weights=weights)))
                                + mlktu)
            # q = q ** (1 / (1 - fi))
            Q = q[:, :, None]
            # v = (Q / Ktu)  ** fi
            v = fi*(Q-np.maximum(lKtu, np.log(reg_K)))

            if (np.any(lKtu == -np.inf)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                # q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(np.exp(qprev)*(1-np.exp(q-qprev))).max()
                err /= max(abs(np.exp(q)).max(), abs(np.exp(qprev)).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))

        if log:
            log['niter'] = i
            log['logu'] = ((u + 1e-300))
            log['logv'] = ((v + 1e-300))
            return np.exp(q), log
        else:
            return np.exp(q)
    else:
        v = np.ones((dimx, dimy, n_hists))/dimx/dimy
        u = np.ones((dimx, dimy))/dimx/dimy
        q = np.ones((dimx, dimy))/dimx/dimy
        err = 1.
        Kx = np.exp(-Cxs/reg)
        Ky = np.exp(-Cys/reg)
        # Kx = np.exp(-Cxs/reg).swapaxes(1,2).swapaxes(0,1) # n_image, dimx, dimx
        # Ky = np.exp(-Cys/reg).swapaxes(1,2).swapaxes(0,1) # n_image, dimy, dimy
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            Kv = prod_sep(Kx, Ky, v)
            # Kv = np.matmul(np.matmul(Kx,v.swapaxes(1,2).swapaxes(0,1)),Ky).swapaxes(0,1).swapaxes(1,2)
            u = (A / (Kv+reg_K)) ** fi
            # Ktu = np.matmul(np.matmul(Kx.swapaxes(1,2),u.swapaxes(1,2).swapaxes(0,1)),Ky.swapaxes(1,2)).swapaxes(0,1).swapaxes(1,2)
            Ktu = prod_sep(Kx.swapaxes(0, 1), Ky.swapaxes(0, 1), u)

            q = ((Ktu ** (1 - fi)).dot(weights))
            q = q ** (1 / (1 - fi))
            Q = q[:, :, None]
            v = (Q / (Ktu+reg_K)) ** fi

            if (np.any(Ktu == 0)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(q-qprev).max()
                err /= max(abs(q).max(), abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))
        if log:
            log['niter'] = i
            log['logu'] = (np.log(u + 1e-300))
            log['logv'] = (np.log(v + 1e-300))
            return q, log
        else:
            return q


def barycenter_sinkhorn2D(A, Cx, Cy, reg, weights=None,
                          numItermax=1000, stopThr=1e-6,
                          verbose=False, log=False, logspace=True,
                          reg_K=1e-16):
    """
    ----------
    A : np.ndarray (dimx,dimy, n_hists)
        `n_hists` training distributions a_i of dimension dimxdim
    Cx : np.ndarray (dimx, dimx)
        x part of separable cost matrix for OT.
    Cy : np.ndarray (dimy, dimy)
        y part of separable cost matrix for OT.
    reg : float
        Entropy regularization term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    logspace : bool, optional
        compuation done in logspace if True
    Returns
    -------
    q : (dim,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.
    """
    dimx, dimy, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[2])
    weights = weights/weights.sum()
    if log:
        log = {'err': []}

    # K = np.exp(- M / reg)

    if logspace:
        v = np.zeros((dimx, dimy, 1))*np.log(1/dimx/dimy)
        u = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        q = np.zeros((dimx, dimy))*np.log(1/dimx/dimy)
        err = 1.
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            lKv = prod_separable_logspace(Cx, Cy, reg, v)
            # u = (A / Kv) ** fi
            u = np.log(A)-np.maximum(lKv, np.log(reg_K))
            # Ktu = K.T.dot(u)
            lKtu = prod_separable_logspace(Cx.T, Cy.T, reg, u)
            q = np.average(lKtu, axis=2, weights=weights)
            Q = q[:, :, None]

            v = (Q-np.maximum(lKtu, np.log(reg_K)))

            if (np.any(lKtu == -np.inf)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                # q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(np.exp(qprev)*(1-np.exp(q-qprev))).max()
                err /= max(abs(np.exp(q)).max(), abs(np.exp(qprev)).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))

        if log:
            log['niter'] = i
            log['logu'] = ((u + 1e-300))
            log['logv'] = ((v + 1e-300))
            return np.exp(q), log
        else:
            return np.exp(q)
    else:
        v = np.ones((dimx, dimy, 1))/dimx/dimy
        u = np.ones((dimx, dimy))/dimx/dimy
        q = np.ones((dimx, dimy))/dimx/dimy
        err = 1.
        Kx = np.exp(-Cx/reg)
        Ky = np.exp(-Cy/reg)
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            Kv = np.tensordot(np.tensordot(Kx, v, axes=([1], [0])),
                              Ky, axes=([1], [0])).swapaxes(1, 2)
            u = (A / (Kv+reg_K))
            Ktu = np.tensordot(np.tensordot(Kx.T, u, axes=([1], [0])),
                               Ky.T, axes=([1], [0])).swapaxes(1, 2)

            q = np.prod(Ktu**weights, axis=2)
            Q = q[:, :, None]
            v = (Q / (Ktu+reg_K))

            if (np.any(Ktu == 0)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(q-qprev).max()
                err /= max(abs(q).max(), abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err))
        if log:
            log['niter'] = i
            log['logu'] = (np.log(u + 1e-300))
            log['logv'] = (np.log(v + 1e-300))
            return q, log
        else:
            return q
