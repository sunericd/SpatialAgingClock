# TEST FILE FOR SPATIAL AGING CLOCKS


# import packages

import spatialclock.deploy # for deploying spatial aging clocks
import spatialclock.proximity # for running proximity effect analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import anndata as ad
import os

# turn off warnings
import warnings
warnings.filterwarnings('ignore')



print("Checking toy data loading...")
try:
    adata = sc.read_h5ad("data/small_data.h5ad")
except:
    raise Exception("Toy data loading failed")

print("Checking age prediction...")
try:
    df = spatialclock.deploy.get_predictions(adata)
except:
    raise Exception("Age prediction failed")
    
print("Checking age acceleration...")
try:
    spatialclock.deploy.get_age_acceleration (adata)
except:
    raise Exception("Age acceleration failed")
    
print("Checking cell proximity...")
try:
    celltypes = pd.unique(adata.obs.celltype).sort_values()

    spatialclock.proximity.nearest_distance_to_celltype(adata,
                             celltype_list=celltypes,
                             sub_id="mouse_id")
except:
    raise Exception("Cell proximity failed")
    
print("Checking proximity effect...")
try:
    cutoff = 30 # this can also be a region-specific dictionary of cutoffs
    celltypes = pd.unique(adata.obs.celltype).sort_values()
    df = spatialclock.proximity.compute_proximity_effects(adata, cutoff, celltypes,
                                                          min_pairs=1)
except:
    raise Exception("Proximity effect failed")