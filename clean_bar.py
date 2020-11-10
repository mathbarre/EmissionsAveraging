#%% Compute Wasserstein barycenter
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import ot
import os
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from dev_unbalanced import barycenter_unbalanced_sinkhorn_dev,barycenter_unbalanced_sinkhorn,barycenter_unbalanced_stabilized,barycenter_unbalanced_stabilized_dev,barycenter_sinkhorn_dev,barycenter_sinkhorn
from numba import njit

#place = "hassirmel"
#place = "pennsylvania"
#place = "pennsylvania"
place = "permian"
#place = "dhakka"
#place="middle_east"

if place == "hassirmel" :
    basepath = ''
elif place == "pennsylvania" :
    basepath = './data/pennsylvania_mines_reprojected/'
elif place == "dhakka" :
    basepath = ''    
elif place == "permian" :
    basepath = './data/permian_wind_reprojected/'    
elif place == "middle_east" :
    basepath = ''    
else :
    basepath = ''

#%% Form data matrix

dimx = 20
dimy= 30
# dimx = 78
# dimy = 87
#dimx = 64
#dimy = 119



quality_thres = 10
cm = 'OrRd'
A = np.zeros((2,1))
wind = np.zeros((2,1))
weight = np.zeros(1)
count_nan = np.zeros((dimx,dimy))
for root, dirs, files in os.walk(basepath):
    for name in files:
        if os.path.splitext(name)[1] == ".tif" or os.path.splitext(name)[1] == ".tiff":
            ptth=os.path.join(root,name)
            dataset = rasterio.open(ptth)
            
            dimx_true,dimy_true = dataset.shape
            data = dataset.read(out_shape=(dataset.count, dimx, dimy), resampling=Resampling.bilinear)
            
            data[0,data[0,:,:]==0]=np.nan
            
            img1 = data[0,:,:] # 2 is the mask
            quality = (data[1,:,:]>= quality_thres).sum()
            data[0,data[1,:,:]<=quality_thres] = np.nan
            dta = np.nan_to_num(data[0,:,:], copy=True, nan=0.0)
            
            
            

            if ((data[1,:,:]>= quality_thres)*dta).sum() >0 :
                w2 = np.average(data[3,:,:],weights=(data[1,:,:]>= quality_thres)*dta) # West-East
                w1 = np.average(data[4,:,:],weights=(data[1,:,:]>= quality_thres)*dta) # South-North
                w1 = -w1 
            else :
                w1 = 0
                w2 = 0
            a=np.clip(data[0,:,:].flatten(),0,np.inf)

            if np.nansum(a) != 0 :
                count_nan = (count_nan + np.isnan(data[0,:,:])*1)
                if (A==0).all():
                    A = a
                    #weight = np.nansum(a)
                    weight = sum(1-(np.isnan(a)))
                else:
                    A=np.vstack((A,a))
                    #weight = np.vstack((weight,np.nansum(a)))
                    weight = np.vstack((weight,sum(1-(np.isnan(a)))))
                if (wind==0).all():
                    wind = np.array([[w1],[w2]])
                else:
                    wind= np.hstack((wind,np.array([[w1],[w2]])))
#A=A/np.nanmax(A.flatten())  
A = A[:,:]
A=A/np.nanmedian(A.flatten()) 
A=np.clip(A,0,10) # CHECK: large outliers
np.nan_to_num(A, copy=False, nan=0.0) # Fill nans with zeros
weight= weight/weight.sum()
wind= wind[:,:]

#%%
@njit
def windMtxLoop(direction,speed,m,n,time=1):
    M = np.zeros((m*n,m*n))
    x=np.column_stack(np.nonzero(np.ones((m,n))))*np.array([dimx_true/dimx,dimy_true/dimy])
    for i in range(m*n):
        for j in range(m*n):
            #WARNING row index corresponds to move on the North/South axis
            proj = direction[0]*(x[j][0]-x[i][0])+direction[1]*(x[j][1]-x[i][1])
            M[i,j] = max(0,(x[j][0]-x[i][0])**2 + (x[j][1]-x[i][1])**2 + speed*time*proj)
    return M

def windMtx(direction,speed,m,n,time=1):
    M = np.zeros((m*n,m*n))
    x=np.column_stack(np.nonzero(np.ones((m,n))))*np.array([dimx_true/dimx,dimy_true/dimy])
    XX =np.array([np.dot(x,direction)]).T
    proj = XX.T-XX
    #WARNING row index corresponds to move on the North/South axis
    M = ot.dist(x) + speed*time*proj
    M = (abs(M)+M)/2
    return M


#%%
#using wasserstein ficher rao metric
@njit
def Mfr(m,n,delta,max_val):
    M = -np.ones((m*n,m*n))
    x=np.column_stack(np.nonzero(np.ones((m,n))))*np.array([dimx_true/dimx,dimy_true/dimy])
    for i in range(m*n):
        for j in range(m*n):
            #WARNING row index corresponds to move on the North/South axis
            if np.sqrt((x[i]-x[j])[0]**2+(x[i]-x[j])[1]**2)/2/delta <= np.pi/2:
                M[i,j] = -np.log(np.cos(np.sqrt((x[i]-x[j])[0]**2+(x[i]-x[j])[1]**2)/2/delta)**2)
    mx = M.max()+1                
    for i in range(m*n):
        for j in range(m*n):
            if M[i,j] == -1:
                M[i,j] = mx    
    return M

delta=15
M_wfr = Mfr(dimx,dimy,delta,1)
M_wfr = M_wfr/M_wfr.max()


#%%


#wind = np.zeros((2,A.shape[0]))
x=np.column_stack(np.nonzero(np.ones((dimx,dimy))))*np.array([dimx_true/dimx,dimy_true/dimy])

M = ot.dist(x)
M = M/ M.max()





M_ = np.zeros((A.shape[1],A.shape[1],1))
M_[:,:,0]= M
Ms = np.zeros((A.shape[1],A.shape[1],A.shape[0]))



wind_factor = 8

#Creates a cost matrix for each image using the wind data
for i in range(A.shape[0]):
    wind_speed = np.linalg.norm(wind[:,i])
    
    if wind_speed > 0 :

        wind_dir = wind[:,i]/wind_speed
        #Mw = windMtx(-wind_dir,wind_speed,dimx,dimy,(i+1)**0.5)
        Mw = windMtxLoop(wind_dir,wind_speed,dimx,dimy,time=wind_factor)
        #Mw = windMtx_wfr(-wind_dir,wind_speed,dimx,dimy,time=wind_factor)

    else : 
        Mw = ot.dist(x)
    
    Ms[:,:,i] = Mw/Mw.max()
    
    
    



#%%

reg=0.0025
reg_m=0.4

G1=barycenter_unbalanced_stabilized_dev(np.transpose(A[:,:]), Ms, reg, reg_m,weights=None,numItermax=500, stopThr=1e-04, verbose=True, log=True,tau=1e17)
G3=ot.barycenter_unbalanced(np.transpose(A[:,:]), M_wfr, reg,reg_m,method='sinkhorn_stabilized',weights=None, numItermax=500, stopThr=1e-04, verbose=True, log=True,tau=1e18)
G2=ot.barycenter_unbalanced(np.transpose(A[:,:]), M, reg,reg_m,method='sinkhorn_stabilized',weights=None, numItermax=500, stopThr=1e-04, verbose=True, log=True,tau=1e18)

imgwind=G1[0].reshape(img1.shape)
img_l2=G2[0][:].reshape(img1.shape)
img_wfr=G3[0][:].reshape(img1.shape)
#imgwind=G1[0].reshape(img1.shape)

# Compute simple mean
imgm=A[:,:].mean(axis=0).reshape(img1.shape)
# Plot result
cm = 'OrRd'
ax1 = plt.subplot(141)
pl.imshow(imgm, cmap=cm)
pl.colorbar()

pl.axis('off')
ax1 = plt.subplot(142)
pl.imshow(img_l2, cmap=cm)
pl.colorbar()

pl.axis('off')
ax1 = plt.subplot(143)
pl.imshow(imgwind, cmap=cm)
pl.colorbar()

pl.axis('off')
ax1 = plt.subplot(144)
pl.imshow(img_wfr, cmap=cm)
pl.colorbar()

pl.axis('off')

pl.show()


# %%