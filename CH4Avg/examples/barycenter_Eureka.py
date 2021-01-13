import numpy as np
import matplotlib.pyplot as plt
from CH4Avg.utils.dev_unbalanced import (barycenter_unbalanced_sinkhorn2D,
                                         barycenter_unbalanced_sinkhorn2D_wind)
from CH4Avg.utils.cost_matrices import CostMtx
from CH4Avg.utils.preprocessing import load_data, preprocess
import csv


basepath = '../data/permian_eureka/'
(A, dimx, dimy, wind, y_over_x_ratio) = load_data(basepath)
Tot = preprocess(A, to_2D=(dimx, dimy))

# Get the mean wind
with open('../data/permian_eureka/quantif.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))


us = np.array([float(data[i][5]) for i in range(1, Tot.shape[2]+1)])
vs = np.array([float(data[i][6]) for i in range(1, Tot.shape[2]+1)])
wind = np.zeros((2, us.shape[0]))
wind[0, :] = -vs
wind[1, :] = us

# Cx of shape dimx ^ 2
(Cx, Cy) = CostMtx(dimx, dimy, y_over_x_ratio=y_over_x_ratio, separable=True)

# For 2D with wind
(Cxs, Cys) = (np.zeros((Cx.shape[0], Cx.shape[0], Tot.shape[2])), np.zeros(
    (Cy.shape[0], Cy.shape[0], Tot.shape[2])))

# negative because we want to "reverse" the effect of the wind
wind_factor = -0.12
for i in range(Tot.shape[2]):
    (Cxs[:, :, i], Cys[:, :, i]) = CostMtx(
        dimx, dimy, y_over_x_ratio=y_over_x_ratio,
        separable=True, wind=wind[:, i],
        wind_factor=wind_factor)

reg = 0.0001
reg_m = 0.5


G2D = barycenter_unbalanced_sinkhorn2D(
    Tot, Cx, Cy, reg, reg_m, weights=Tot.sum(axis=0).sum(axis=0)/Tot.sum(),
    numItermax=400, stopThr=1e-4, verbose=True,
    log=True, logspace=False, reg_K=1e-16)
G2Dw = barycenter_unbalanced_sinkhorn2D_wind(
    Tot, Cxs, Cys, reg, reg_m, weights=Tot.sum(axis=0).sum(axis=0)/Tot.sum(),
    numItermax=400, stopThr=1e-4, verbose=True, log=True, logspace=False,
    reg_K=1e-16)
# %%
cm = 'OrRd'
ax1 = plt.subplot(131)
plt.imshow(Tot.mean(axis=2), cmap=cm)
plt.title("arithmetic mean")
ax1 = plt.subplot(132)
plt.imshow(G2D[0], cmap=cm)
plt.title("2D Wasserstein")
ax1 = plt.subplot(133)
plt.imshow(G2Dw[0], cmap=cm)
plt.title("2D Wasserstein + wind")
plt.show()
