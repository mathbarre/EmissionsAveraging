import os

import rasterio
from rasterio.warp import reproject, Resampling

place = "pennsylvania_mines"
basepath = '../data/'+place

dirName = '../data/'+place+'_reprojected/'

if not os.path.exists(dirName):
    os.mkdir(dirName)
    first = True

    for root, dirs, files in os.walk(basepath):
        for name in files:
            if os.path.splitext(name)[1] == ".tif" or\
             os.path.splitext(name)[1] == ".tiff":
                ptth = os.path.join(root, name)
                dataset = rasterio.open(ptth)
                if first:
                    ref_profile = dataset.meta.copy()
                    first = False
                with rasterio.open(dirName + os.path.splitext(name)[0]+'_reprojected'+os.path.splitext(name)[1], 'w', **ref_profile) as dst:
                    for i in range(1, dataset.count+1):
                        reproject(source=rasterio.band(dataset, i), src_transform=dataset.transform, src_crs=dataset.crs, dst_crs=dst.crs, dst_transform=dst.transform, destination=rasterio.band(dst, i), resampling=Resampling.bilinear)
    print(place+" has been reprojected")
else:
    print("Seems that "+place+" is already reprojected")
