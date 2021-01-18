from CH4Avg.utils.dev_unbalanced import (barycenter_unbalanced_sinkhorn2D,
                                         barycenter_unbalanced_sinkhorn2D_wind)
from CH4Avg.utils.cost_matrices import CostMtx
from CH4Avg.utils.preprocessing import load_data, preprocess
import ot
import numpy as np
import matplotlib.pyplot as plt


def gauss_cloud(centers, sigmax, sigmay, dimx, dimy, nb_pts):
    img = np.zeros((dimx, dimy))
    for center in centers:
        gauss = np.vstack((sigmax*np.random.randn(1, nb_pts), sigmay*np.random.randn(1, nb_pts)))
        for i in range(nb_pts):
            if np.int(np.floor(gauss[0,i]))+center[0] < dimx and (np.int(np.floor(gauss[1, i]))+center[1]) < dimy :
                img[np.int(np.floor(gauss[0,i]))+center[0], np.int(np.floor(gauss[1, i]))+center[1]] += 1
    return img/len(centers)


dimx = 100
dimy = 100
n=100000

a =(gauss_cloud(np.array([[20,20]]),10,10, dimx, dimy,n))/n
b =(gauss_cloud(np.array([[80,80]]),10,10, dimx, dimy,2*n))/n
A = np.dstack((a[:,:,None],b[:,:,None]))

(Cx, Cy) = CostMtx(dimx, dimy, separable=True)

G2D = barycenter_unbalanced_sinkhorn2D(
    A, Cx, Cy, 0.0001, 0.08, weights=A.sum(axis=0).sum(axis=0)/A.sum(),
    numItermax=500, stopThr=1e-7, verbose=True,
    log=True, logspace=True, reg_K=0)

# M = CostMtx(dimx, dimy)
# B = np.vstack((a.flatten(),b.flatten()))
# Gtest = ot.barycenter_unbalanced(
#     B.T, M, 0.001, 0.05, weights=None, method="sinkhorn",
#     numItermax=300, stopThr=1e-8, verbose=True,
#     log=True)

plt.imshow(G2D[0])
plt.show()

# plt.imshow(Gtest[0].reshape((dimx,dimy)))
# plt.show()
plt.imshow(a+b)
