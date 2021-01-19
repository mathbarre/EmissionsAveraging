import rasterio
import numpy as np
import os


def load_data(ptth, quality_thr=None, iswind=None, window=None):
    """
    Input
    -pthh : String, path to the folder containing the tiff images
    -quality_thr: (optional), float, qualitity threshold for the pixels,
    assumed if it appears, it is at channel 1
    -iswind: (optional), (int,int), channels of the wind (W-E,S-N)

    Output
    -A the flatten time series A of dimension (n_images,(dimx*dimy))
    -dimx, int
    -dimy, int
    -wind, None or array of dimension (2,n_images)
    -y_over_x_ration, float
    """
    A = None
    wind = None
    for root, dirs, files in os.walk(ptth):
        for name in files:
            if os.path.splitext(name)[1] == ".tif" or\
             os.path.splitext(name)[1] == ".tiff":
                ptth = os.path.join(root, name)
                dataset = rasterio.open(ptth)
                # x latitude, y longitude
                y_over_x_ratio = abs(dataset.transform[0]/dataset.transform[4])
                data = dataset.read(window=window)
                dimx, dimy = data.shape[1:]
                if not(quality_thr is None):
                    data[0, data[1, :, :] <= quality_thr] = np.nan
                a = np.clip(data[0, :, :].flatten(), 0, np.inf)
                if np.nansum(a) != 0:
                    w = None
                    if not(iswind is None):
                        # get wind info, can add refinement on how the mean
                        # wind is computed
                        w2 = data[iswind[0], :, :].mean()
                        w1 = -data[iswind[1], :, :].mean()
                        w = np.array([[w1], [w2]])
                    if A is None:
                        A = a
                        if not(iswind is None):
                            wind = w
                    else:
                        A = np.vstack((A, a))
                        if not(iswind is None):
                            wind = np.hstack((wind, w))
    return A, dimx, dimy, wind, y_over_x_ratio


def preprocess(A, normalize=(lambda x: x), clip=np.inf, to_2D=None):
    # Super basic for now
    """
    Input:
    -A, time series of dimension (n_images,dimx*dimy)
    -normalize, (optional) normalization function
    -clip, (optional) clip the value above
    -to_2D, (optional) (int,int), output is of shape (dimx,dimy,n_images)
    """
    A = normalize(A)
    A = np.clip(A, 0, clip)  # CHECK: large outliers
    np.nan_to_num(A, copy=False, nan=0.0)
    if not(to_2D is None):
        Tot = A.reshape((A.shape[0], to_2D[0], to_2D[1]))
        Tot = Tot.swapaxes(0, 1).swapaxes(1, 2)
        return Tot
    return A


def dummyfies(A):
    if A.ndim == 3:
        (dimx, dimy, n_images) = A.shape
        res = np.zeros(dimx+1, dimy+1, n_images)
        masses = A.sum(axis=0).sum(axis=0)
        masses -= masses.max()
        masses *= -1
        res[:-1, :-1, :] = A
        res[-1, :-1, :] = masses/(dimx+dimy)
        res[:-1, -1, :] = masses/(dimx+dimy)
    else:
        (n_images, dim) = A.shape
        res = np.zeros((n_images, dim+1))
        masses = np.nansum(A, axis=1)
        masses -= masses.max()
        masses *= -1
        res[:, :-1] = A
        res[:, -1] = masses
    return res
