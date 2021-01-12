#%% Code to align different images in a new dataset, given an image for coordinate reference
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import ot
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling




place = "pennsylvania_mines_reprojected"
basepath = '../data/'+place
ref_dataset = rasterio.open(basepath+'/s5p_pennsylvania_mines_2019-01-04T17:57:03Z_reprojected.tiff')

dirName = '../data/'+place+'_reprojected/'

if not os.path.exists(dirName):
    os.mkdir(dirName)

    

#%%
ref_profile = ref_dataset.meta.copy()

for root, dirs, files in os.walk(basepath):
    for name in files:
        if os.path.splitext(name)[1] == ".tif" or os.path.splitext(name)[1] == ".tiff":
            ptth=os.path.join(root,name)
            dataset = rasterio.open(ptth)
            #with rasterio.open('/Users/mathieubarre/Dropbox/Averaging/data/ch4_crops_with_wind/'+place+'_reprojected/'+os.path.splitext(name)[0]+'_reprojected'+os.path.splitext(name)[1], 'w', **ref_profile) as dst : 
            with rasterio.open(dirName+os.path.splitext(name)[0]+'_reprojected'+os.path.splitext(name)[1], 'w', **ref_profile) as dst : 
                for i in range(1,dataset.count+1):
                #for i in range(1,3):
                    reproject(source=rasterio.band(dataset,i),src_transform=dataset.transform,src_crs=dataset.crs,dst_crs=dst.crs,dst_transform=dst.transform,destination = rasterio.band(dst,i),resampling=Resampling.bilinear)
                
 





# %%
