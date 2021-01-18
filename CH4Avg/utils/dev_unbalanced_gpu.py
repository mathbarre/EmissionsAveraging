# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers 
Run on GPU
"""


import cupy as cp


def prod_separable_logspace(Cx, Cy, gamma, v):
    """
    implementation of Algorithm 3 of 
    Wasserstein Dictionary Learning: Optimal Transport-based unsupervised non-linear dictionary learning
    (Morgan Schmitz, Matthieu Heitz, Nicolas Bonneel, Fred Maurice Ngolè Mboula, David Coeurjolly, Marco Cuturi, Gabriel Peyré, Jean-Luc Starck)
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
    R = cp.zeros(v.shape)
    for i in range(n_hist):
        if Cy.ndim > 2:
            x = -Cy[:, :, i]/gamma + v[:, None, :, i]
        else:
            x = -Cy[:, :]/gamma + v[:, None, :, i]
        mx = cp.max(x, axis=2)
        mx[mx == -cp.inf] = 0
        A = cp.log(cp.exp(x-mx[:, :, None]).sum(axis=2))+mx
        if Cx.ndim > 2:
            y = -Cx[:, None, :, i]/gamma + A.T
        else:
            y = -Cx[:, None, :]/gamma + A.T
        my = cp.max(y, axis=2)
        my[my == -cp.inf] = 0
        R[:, :, i] = cp.log(cp.exp(y-my[:, :, None]).sum(axis=2))+my
    return R


def barycenter_unbalanced_sinkhorn2D_gpu(A, Cx, Cy, reg, reg_m, weights=None,
                                         numItermax=1000, stopThr=1e-6,
                                         verbose=False, log=False,
                                         logspace=False, reg_K=1e-16):
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
        weights = cp.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[2])
    weights = cp.asarray(weights/weights.sum())
    if log:
        log = {'err': []}

    fi = reg_m / (reg_m + reg)
    if logspace:
        v = cp.zeros((dimx, dimy, 1))*cp.log(1/dimx/dimy)
        u = cp.zeros((dimx, dimy))*cp.log(1/dimx/dimy)
        q = cp.zeros((dimx, dimy))*cp.log(1/dimx/dimy)
        err = 1.
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            lKv = prod_separable_logspace(Cx, Cy, reg, v)
            u = fi*(cp.log(A)-cp.maximum(lKv, cp.log(reg_K)))
            lKtu = prod_separable_logspace(Cx.T, Cy.T, reg, u)
            mlktu = (1-fi)*cp.max(lKtu, axis=2)
            q = (1 / (1 - fi))*((cp.log(cp.average(cp.exp((1-fi)*lKtu -
                                                   mlktu[:, :, None]),
                                                   axis=2, weights=weights)))
                                + mlktu)
            Q = q[:, :, None]
            v = fi*(Q-cp.maximum(lKtu, cp.log(reg_K)))

            if (cp.any(lKtu == -cp.inf)
                    or cp.any(cp.isnan(u)) or cp.any(cp.isnan(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                # q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(cp.exp(qprev)*(1-cp.exp(q-qprev))).max()
                err /= max(abs(cp.exp(q)).max(), abs(cp.exp(qprev)).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err.get()))

        if log:
            log['niter'] = i
            log['logu'] = ((u + 1e-300))
            log['logv'] = ((v + 1e-300))
            return cp.exp(q), log
        else:
            return cp.exp(q)
    else:
        v = cp.ones((dimx, dimy, 1))/dimx/dimy
        u = cp.ones((dimx, dimy))/dimx/dimy
        q = cp.ones((dimx, dimy))/dimx/dimy
        err = 1.
        Kx = cp.exp(-Cx/reg)
        Ky = cp.exp(-Cy/reg)
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            Kv = cp.tensordot(cp.tensordot(Kx, v, axes=([1], [0])),
                              Ky, axes=([1], [0])).swapaxes(1, 2)
            u = cp.power(cp.divide(A, (Kv+reg_K)), fi)
            Ktu = cp.tensordot(cp.tensordot(cp.transpose(Kx), u, axes=([1], [0])),
                               cp.transpose(Ky), axes=([1], [0])).swapaxes(1, 2)
            q = cp.power(cp.dot(cp.power(Ktu, (1 - fi)), weights), (1 / (1 - fi)))
            v = cp.power(cp.divide(q[:, :, None], (Ktu+reg_K)), fi)

            if (cp.any(Ktu == 0)
                    or cp.any(cp.isnan(u)) or cp.any(cp.isnan(v))
                    or cp.any(cp.isinf(u)) or cp.any(cp.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = cp.abs(q-qprev).max()
                err /= max(cp.abs(q).max(), cp.abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err.get()))
        if log:
            log['niter'] = i
            log['logu'] = (cp.log(u + 1e-300))
            log['logv'] = (cp.log(v + 1e-300))
            return q, log
        else:
            return q


def barycenter_unbalanced_sinkhorn2D_wind_gpu(A, Cxs, Cys, reg, reg_m,
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
        weights = cp.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[2])
    weights = cp.asarray(weights/weights.sum())
    if log:
        log = {'err': []}

    fi = reg_m / (reg_m + reg)
    if logspace:
        v = cp.zeros((dimx, dimy, n_hists))*cp.log(1/dimx/dimy)
        u = cp.zeros((dimx, dimy))*cp.log(1/dimx/dimy)
        q = cp.zeros((dimx, dimy))*cp.log(1/dimx/dimy)
        err = 1.
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            lKv = prod_separable_logspace(Cxs, Cys, reg, v)
            u = fi*(cp.log(A)-cp.maximum(lKv, cp.log(reg_K)))
            lKtu = prod_separable_logspace(Cxs.swapaxes(0, 1),
                                           Cys.swapaxes(0, 1), reg, u)
            mlktu = (1-fi)*cp.max(lKtu, axis=2)
            q = (1 / (1 - fi))*((cp.log(cp.average(cp.exp((1-fi)*lKtu
                                                          - mlktu[:, :, None]),
                                                   axis=2, weights=weights)))
                                + mlktu)
            Q = q[:, :, None]
            v = fi*(Q-cp.maximum(lKtu, cp.log(reg_K)))

            if (cp.any(lKtu == -cp.inf)
                    or cp.any(cp.isnan(u)) or cp.any(cp.isnan(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                #q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = abs(cp.exp(qprev)*(1-cp.exp(q-qprev))).max()
                err /= max(abs(cp.exp(q)).max(), abs(cp.exp(qprev)).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err.get()))

        if log:
            log['niter'] = i
            log['logu'] = ((u + 1e-300))
            log['logv'] = ((v + 1e-300))
            return cp.exp(q), log
        else:
            return cp.exp(q)
    else:
        v = cp.ones((dimx, dimy, 1))/dimx/dimy
        u = cp.ones((dimx, dimy))/dimx/dimy
        q = cp.ones((dimx, dimy))/dimx/dimy
        err = 1.
        Kx = cp.exp(-Cxs/reg).swapaxes(1, 2).swapaxes(0, 1)  # n_image, dimx, dimx
        Ky = cp.exp(-Cys/reg).swapaxes(1, 2).swapaxes(0, 1)  # n_image, dimy, dimy
        for i in range(numItermax):
            uprev = u.copy()
            vprev = v.copy()
            qprev = q.copy()

            Kv = cp.matmul(cp.matmul(Kx, v.swapaxes(1, 2).swapaxes(0, 1)),
                        Ky).swapaxes(0, 1).swapaxes(1, 2)
            u = cp.power(cp.divide(A, (Kv+reg_K)), fi)
            Ktu = cp.matmul(cp.matmul(Kx.swapaxes(1, 2),
                                    u.swapaxes(1, 2).swapaxes(0, 1)),
                            Ky.swapaxes(1, 2)).swapaxes(0, 1).swapaxes(1, 2)
            q = cp.power(cp.dot(cp.power(Ktu, (1 - fi)), weights), (1 / (1 - fi)))
            v = cp.power(cp.divide(q[:, :, None], (Ktu+reg_K)), fi)

            if (cp.any(Ktu == 0)
                    or cp.any(cp.isnan(u)) or cp.any(cp.isnan(v))
                    or cp.any(cp.isinf(u)) or cp.any(cp.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Numerical errors at iteration %s' % i)
                u = uprev
                v = vprev
                q = qprev
                break
                # compute change in barycenter
            if (i % 10 == 0) or i == 0:
                err = cp.abs(q-qprev).max()
                err /= max(cp.abs(q).max(), cp.abs(qprev).max(), 1.)
                if log:
                    log['err'].append(err)
                # if barycenter did not change + at least 10 iterations - stop
                if err < stopThr and i > 10:
                    break

                if verbose:
                    if i % 100 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(i, err.get()))
        if log:
            log['niter'] = i
            log['logu'] = (cp.log(u + 1e-300))
            log['logv'] = (cp.log(v + 1e-300))
            return q, log
        else:
            return q
