import numpy as np
from numba import njit


def CostMtx(dimx, dimy, y_over_x_ratio=1, cost="euclidean", separable=False, wind=None, wind_factor=0,delta=0):
    # TODO add doc
    """
    Construct the cost matrix need for the transport.
    Given 2 points P1 P2 on a 2D grid, the cost to go from P1 to P2
    c(P1,P2) is smaller if the wind blows from P1 to P2.
    In our barycenter problem since we want to "reverse" the effect of the wind,
     we consider a negative wind factor.
    """
    if cost == "euclidean":
        if not(separable):
            x = np.column_stack(np.nonzero(np.ones((dimx, dimy)))) \
                * np.array([1, y_over_x_ratio])
            M = (x[:, 0, None]-x[:, 0]) ** 2 + (x[:, 1, None]-x[:, 1]) ** 2
            if not(wind is None):
                X = np.dot(x, wind)
                M += wind_factor*(X[:, None]-X)
            return M/M.max()
        else:
            xs = np.arange(dimx)
            ys = np.arange(dimy)
            Cx = np.array(xs[:, None]-xs, dtype=np.float)**2
            Cy = np.array(ys[:, None]-ys, dtype=np.float)**2
            if not(wind is None):
                Cx += wind_factor*wind[0]*(xs[:, None]-xs)*np.abs((xs[:, None]-xs))
                Cy += wind_factor*wind[1]*(ys[:, None]-ys)*np.abs((ys[:, None]-ys))
            mx = Cx.max()+Cy.max()
            return (Cx/mx, Cy/mx)
    if cost == "wfr":
        return Costwfr(dimx, dimy, delta)
    return "cost not implemented yet"


# Can be made more efficientt
@njit
def Costwfr(dimx, dimy, delta, y_over_x_ratio):
    M = -np.ones((dimx*dimy, dimx*dimy))
    x = np.column_stack(np.nonzero(np.ones((dimx, dimy)))) \
        * np.array([1, y_over_x_ratio])
    for i in range(dimx*dimy):
        for j in range(dimx*dimy):
            # WARNING row index corresponds to move on the North/South axis
            if np.sqrt((x[i]-x[j])[0]**2+(x[i]-x[j])[1]**2)/2/delta <= np.pi/2:
                M[i, j] = -np.log(np.cos(np.sqrt((x[i]-x[j])[0]**2+(x[i]-x[j])[1]**2)/2/delta)**2)
    mx = M.max()+1
    for i in range(dimx*dimy):
        for j in range(dimx*dimy):
            if M[i, j] == -1:
                M[i, j] = mx
    return M/M.max()
