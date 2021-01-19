# EmissionsAveraging
## Requirements
- POT (https://github.com/PythonOT/POT)
- rasterio (https://rasterio.readthedocs.io/en/latest/)
- numba (https://numba.readthedocs.io/en/stable/user/installing.html)
- Cupy (https://cupy.dev/) (for gpu)

To install the package do:

    cd EmissionsAverging/

    pip install -e .

Then you can run wasserstein barycenter on the Permian bassin by doing
    
    cd CH4Avg/examples/
    
    python barycenter_sentinelS5P.py
   
 If Cupy is installed and you have a cuda compatible gpu on your machine
 you can run wasserstein barycenter with Eureka data by doing
      
     cd CH4Avg/examples/gpu/
     
     python barycenter_Eureka_gpu.py
    
