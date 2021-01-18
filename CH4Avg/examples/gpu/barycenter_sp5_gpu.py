from CH4Avg.utils.dev_unbalanced import (barycenter_unbalanced_sinkhorn2D,
                                         barycenter_unbalanced_sinkhorn2D_wind)
from CH4Avg.utils.dev_unbalanced_gpu import (
 barycenter_unbalanced_sinkhorn2D_gpu,
 barycenter_unbalanced_sinkhorn2D_wind_gpu)
from CH4Avg.utils.cost_matrices import CostMtx
from CH4Avg.utils.preprocessing import load_data, preprocess
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time


basepath = '../../data/permian_wind_reprojected/'

(A, dimx, dimy, wind, y_over_x_ratio) = load_data(
    basepath, quality_thr=1, iswind=(3, 4))
A = preprocess(A[:100, :], normalize=(lambda x: x/np.nanmean(x)))
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
wind_factor = -0.15
for i in range(Tot.shape[2]):
    (Cxs[:, :, i], Cys[:, :, i]) = CostMtx(
        dimx, dimy, y_over_x_ratio=y_over_x_ratio,
        separable=True, wind=wind[:, i],
        wind_factor=wind_factor)

reg = 0.0015
reg_m = 0.5

start_time = time.time()
G2D = barycenter_unbalanced_sinkhorn2D(
    Tot, Cx, Cy, reg, reg_m, weights=None,
    numItermax=100, stopThr=1e-4, verbose=True,
    log=True, logspace=True, reg_K=1e-16)
print(time.time() - start_time, "s cpu")
start_time = time.time()
G2Dw = barycenter_unbalanced_sinkhorn2D_wind(
    Tot, Cxs, Cys, reg, reg_m, weights=None, numItermax=100,
    stopThr=1e-4, verbose=True, log=True, logspace=True, reg_K=1e-16)
print(time.time() - start_time, "s cpu")

Tot_gpu = cp.array(Tot)
(Cx_gpu, Cy_gpu) = (cp.array(Cx), cp.array(Cy))
(Cxs_gpu, Cys_gpu) = (cp.array(Cxs), cp.array(Cys))

start_time = time.time()
G2D_gpu = barycenter_unbalanced_sinkhorn2D_gpu(
    Tot_gpu, Cx_gpu, Cy_gpu, reg, reg_m, weights=None,
    numItermax=300, stopThr=1e-4, verbose=True,
    log=True, logspace=True, reg_K=1e-16)
print(time.time() - start_time, "s gpu")
start_time = time.time()
G2Dw_gpu = barycenter_unbalanced_sinkhorn2D_wind_gpu(
    Tot_gpu, Cxs_gpu, Cys_gpu, reg, reg_m, weights=None, numItermax=300,
    stopThr=1e-4, verbose=True, log=True, logspace=True, reg_K=1e-16)
print(time.time() - start_time, "s gpu")
# %%
cm = 'OrRd'
ax1 = plt.subplot(231)
plt.imshow(Tot.mean(axis=2), cmap=cm)
plt.title("arithmetic mean")
ax1 = plt.subplot(232)
plt.imshow(G2D[0], cmap=cm)
plt.title("2D Wasserstein")
ax1 = plt.subplot(233)
plt.imshow(G2Dw[0], cmap=cm)
plt.title("2D Wasserstein + wind")
ax1 = plt.subplot(234)
plt.imshow(G2D_gpu[0].get(), cmap=cm)
plt.title("2D Wasserstein gpu")
ax1 = plt.subplot(235)
plt.imshow(G2Dw_gpu[0].get(), cmap=cm)
plt.title("2D Wasserstein + wind gpu")
plt.savefig("./test.png")
# %%