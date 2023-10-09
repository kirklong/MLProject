# Clustering / ARM code and data

This folder contains the code needed to perform ARM and clustering methods for the project along with versions of the data prepared specifically for these methods.

The code [`genBasketData.py`](genBasketData.py) takes in the main datasets and creates versions of the original data ([MCMCbasket.csv](MCMCbasket.csv), [SDSS_noSizeBasket.csv](SDSS_noSizeBasket.csv), [SDSS_wSizeBasket.csv](SDSS_wSizeBasket.csv)) that are in the transaction basket format for analysis with [`ARM.r`](ARM.r). 

The code [`genClusterData.py`](genClusterData.py) takes in the main datasets and creates versions of the original data suitable for clustering, splitting them up by object/model type. These are then combined in [`kmeans.py`](kmeans.py), with the data further normalized and k-means applied there as well. The hierarchical clustering is done with [`hclust.r`](hclust.r), which uses the combined datasets and again normalizes but applies the hclust algorithm. The full combined dataset is only available for the disk-wind model due to size limitations, but samples of both SDSS datasets are available as well.
