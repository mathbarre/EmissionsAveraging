# EmissionsAveraging
## Requirements
- POT (https://github.com/PythonOT/POT)
- rasterio (https://rasterio.readthedocs.io/en/latest/)
- numba (https://numba.readthedocs.io/en/stable/user/installing.html)

To install the package do:

    cd EmissionsAverging/

    pip install -e .

Then you can run wasserstein barycenter on the Permian bassin by doing
    
    cd CH4Abg/examples/
    
    python barycenter_sentinelS5P.py