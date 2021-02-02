from CH4Avg.utils.dev_unbalanced import (barycenter_unbalanced_sinkhorn2D,
                                         barycenter_unbalanced_sinkhorn2D_wind)
from CH4Avg.utils.cost_matrices import CostMtx
from CH4Avg.utils.preprocessing import load_data, preprocess
import ot
import numpy as np
import matplotlib.pyplot as plt
import time

basepath = '../data/permian_wind_reprojected/'

(A, dimx, dimy, wind, y_over_x_ratio) = load_data(
    basepath, quality_thr=1, iswind=(3, 4))
A = preprocess(A[:, :], normalize=(lambda x: x/np.nanmean(x)))
Tot = preprocess(A, to_2D=(dimx, dimy))

# M is of shape shape dimx / dimy ^ 4
M = CostMtx(dimx, dimy, y_over_x_ratio=y_over_x_ratio)
# Same as M above, but separable (hence much faster)
# Cx of shape dimx ^ 2
(Cx, Cy) = CostMtx(dimx, dimy, y_over_x_ratio=y_over_x_ratio, separable=True)

# For 2D with wind
(Cxs, Cys) = (np.zeros((Cx.shape[0], Cx.shape[0], Tot.shape[2])), np.zeros(
    (Cy.shape[0], Cy.shape[0], Tot.shape[2])))

# negative because we want to "reverse" the effect of the wind
wind_factor = 5
for i in range(Tot.shape[2]):
    (Cxs[:, :, i], Cys[:, :, i]) = CostMtx(
        dimx, dimy, y_over_x_ratio=y_over_x_ratio,
        separable=True, wind=wind[:, i],
        wind_factor=wind_factor, cost="wind_new")

reg = 0.0025
reg_m = 0.5

Gtest = ot.barycenter_unbalanced(
    A.T, M, reg, reg_m, method="sinkhorn_stabilized",
    stopThr=1e-4, log=True, verbose=True, tau=1e18)
start_time = time.time()
G2D = barycenter_unbalanced_sinkhorn2D(
    Tot, Cx, Cy, reg, reg_m, weights=None,
    numItermax=300, stopThr=1e-4, verbose=True,
    log=True, logspace=False, reg_K=1e-16)
print(time.time()-start_time)
G2Dw = barycenter_unbalanced_sinkhorn2D_wind(
    Tot, Cxs, Cys, reg, reg_m, weights=None, numItermax=300,
    stopThr=1e-4, verbose=True, log=True, logspace=False, reg_K=1e-16)
# %%
cm = 'OrRd'
ax1 = plt.subplot(141)
plt.imshow(Tot.mean(axis=2), cmap=cm)
plt.title("arithmetic mean")
ax1 = plt.subplot(142)
plt.imshow(Gtest[0].reshape((dimx, dimy)), cmap=cm)
plt.title("POT 1D Wasserstein")
ax1 = plt.subplot(143)
plt.imshow(G2D[0], cmap=cm)
plt.title("2D Wasserstein")
ax1 = plt.subplot(144)
plt.imshow(G2Dw[0], cmap=cm)
plt.title("2D Wasserstein + wind")
plt.show()
# %%
