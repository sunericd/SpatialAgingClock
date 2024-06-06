import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.neighbors import BallTree

from .utils import *


def nearest_distance_to_celltype(adata, celltype_list, sub_id=None):
    '''
    For each cell in adata, computes the distance to the nearest cell of the given cell types and adds them to adata.obs
    as adata.obs['{celltype}_nearest_distance']
    
        adata [AnnData] - anndata containing the spatial transcriptomics and has adata.obs['celltype'] and adata.obsm['spatial']
        celltype_list [list of str] - celltype string identifiers found in adata.obs['celltype']
        sub_id [str or None] - name of adata.obs column to use to subset before identifying distances
    
    Uses Ball-Tree algorithm
    
    Adds nearest distances to adata.obs[f'{ct}_nearest_distance']
    '''
    for ct in celltype_list:
        
        # Subset if needed
        if sub_id is None:
        
            # Subset into spatial coordinates
            spatial_targets = adata[adata.obs["celltype"]==ct].obsm["spatial"].copy()
            spatial_all = adata.obsm["spatial"].copy()

            # Create a BallTree 
            tree = BallTree(spatial_targets, leaf_size=2)

            # Query the BallTree
            distances2, idxs = tree.query(spatial_all,  k=2)
            # replace self distances of zero with next closest
            distances = distances2[:,0].copy()
            secondary_distances = distances2[:,1].copy()
            distances[distances==0] = secondary_distances[distances==0] 

            # Update results
            adata.obs[f'{ct}_nearest_distance'] = distances
            
        else:
            all_distances = np.ones(adata.shape[0])*np.nan
            
            for sid in np.unique(adata.obs[sub_id]):
                
                sub_adata = adata[adata.obs[sub_id]==sid]
                
                # Subset into spatial coordinates
                spatial_targets = sub_adata[sub_adata.obs["celltype"]==ct].obsm["spatial"].copy()
                if spatial_targets.shape[0] < 2:
                    continue
                spatial_all = sub_adata.obsm["spatial"].copy()

                # Create a BallTree 
                tree = BallTree(spatial_targets, leaf_size=2)

                # Query the BallTree
                distances2, idxs = tree.query(spatial_all,  k=2)
                # replace self distances of zero with next closest
                distances = distances2[:,0].copy()
                secondary_distances = distances2[:,1].copy()
                distances[distances==0] = secondary_distances[distances==0] 
                
                all_distances[adata.obs[sub_id]==sid] = distances.flatten()

            # Update results
            adata.obs[f'{ct}_nearest_distance'] = all_distances

def compute_proximity_effects (adata, cutoff, celltypes, cutoff_multiplier=1, ring_width=None, region_obs="region", celltype_obs="celltype", animal_obs="mouse_id", comparison="farthest", min_pairs=50):
    '''
    Compute proximity stats
    
        cutoff [int or dict] - if int, then use this cutoff distance for all examples
                             - if dict, then use {cutoff[region] = int} distance 
        celltypes [lst] - list of strings specifying effector cell types to compute proximity effects for
        cutoff_multiplier [float] - multipler for radius cutoff
        ring_width [None or float] - width of ring to sample near cells where outer distance is cutoff*cutoff_multipler; if None, then sample all cells within cutoff (i.e. circle)
        region_obs [str] - key in adata.obs to get region labels
        celltype_obs [str] - key in adata.obs to get cell type labels
        animal_obs [str] - key in adata.obs to get animal/sample labels
        comparison [str] - how to determine "far" comparison group ("farthest", "random", "transcript_count")
        min_pairs [int] - minimum number of cells in near/far set to compute proximity effect for
    
    Returns:
        comb_df [Dataframe] - containing the following columns:
            "Near Cell", effector cell type name
            "AgeAccel Cell", target cell type name
            "n", cutoff multiplier used
            "t", t-test statistic
            "p", p-value from t-test
            "Aging Effect", Proximity Effect
            "Near Freq", normalized frequency of interactions
            "Near Num", number of interactions
    '''
    comb_df = pd.DataFrame([])

    for celltype in celltypes:
        
        for ct in pd.unique(adata.obs.celltype).sort_values():
            
            sub_adata = adata[adata.obs["celltype"]==ct]
            
            # get paired proximity sets
            near_ages = get_paired_proximity_labels(sub_adata, cutoff, celltype,
                                                    cutoff_multiplier=cutoff_multiplier,
                                                    ring_width=ring_width,
                                                    region_obs=region_obs,
                                                    celltype_obs=celltype_obs,
                                                    animal_obs=animal_obs,
                                                    comparison=comparison)
            
            # run test
            df = get_stats_df(sub_adata, near_ages, cutoff_multiplier, celltype, ct,
                              min_pairs=min_pairs)
            
            comb_df = pd.concat((comb_df, df))
            
    return (comb_df)


            
def get_paired_proximity_labels (adata, cutoff, celltype, cutoff_multiplier=1, ring_width=None, region_obs="region", celltype_obs="celltype", animal_obs="mouse_id", comparison="farthest"):
    '''
    Returns proximity labels for cell types based on cutoff distance
    
        cutoff [int or dict] - if int, then use this cutoff distance for all examples
                             - if dict, then use {cutoff[region] = int} distance                     
        celltype [str] - effector cell type to compute proximity relation to
        cutoff_multiplier [float] - multipler for radius cutoff
        ring_width [None or float] - width of ring to sample near cells where outer distance is cutoff*cutoff_multipler; if None, then sample all cells within cutoff (i.e. circle)
        region_obs [str] - key in adata.obs to get region labels
        celltype_obs [str] - key in adata.obs to get cell type labels
        animal_obs [str] - key in adata.obs to get animal/sample labels
        comparison [str] - how to determine "far" comparison group ("farthest", "random", "transcript_count")
        
    Returns:
        prox_labels [arr] - array of strings ("Near" or "Far" or "Other") specifying proximity label for each cell
    '''
    
    # init labels
    prox_labels = np.array(["Other"]*adata.shape[0])
    
    # region-based labeling
    for mouse in np.unique(adata.obs[animal_obs]):
        
        sub_adata = adata[adata.obs[animal_obs]==mouse].copy()
        mouse_bool = (adata.obs[animal_obs]==mouse)
        sub_prox_labels = np.array(["Other"]*sub_adata.shape[0])
        
        for region in np.unique(sub_adata.obs[region_obs]):
            
            # get distance cutoff for "near"
            if isinstance(cutoff, int):
                cutoff_dist = cutoff * cutoff_multiplier
            else:
                cutoff_dist = cutoff[region] * cutoff_multiplier
            
            # get "near" cells
            if ring_width is None: # regular approach
                near_bool = (sub_adata.obs.region==region)&(sub_adata.obs[f"{celltype}_nearest_distance"] < cutoff_dist)
            else: # area-restricted approach
                near_bool = (sub_adata.obs.region==region)&(sub_adata.obs[f"{celltype}_nearest_distance"] < cutoff_dist)&(sub_adata.obs[f"{celltype}_nearest_distance"] > cutoff_dist-ring_width)
            num_near = np.sum(near_bool)
            
            # get "far" cells
            if num_near > 0: # takes care of edge case where num_near==0
                try:
                    region_idxs = np.arange(sub_adata.shape[0])[(sub_adata.obs.region==region)&(sub_adata.obs[f"{celltype}_nearest_distance"] >= cutoff_dist)]
                    region_dists = np.array(sub_adata.obs[f"{celltype}_nearest_distance"])[region_idxs]
                    
                    if comparison == "farthest":
                        farthest_in_region = np.argpartition(region_dists,-num_near)[-num_near:]
                        far_idxs = region_idxs[farthest_in_region]    
                    elif comparison == "random":
                        np.random.seed(444)
                        far_idxs = np.random.choice(region_idxs, num_near, replace=False)
                    elif comparison == "transcript_count":
                        near_mean_transcript_count = sub_adata[near_bool].obs.transcript_count.mean()
                        region_counts = np.array(sub_adata.obs.transcript_count)[region_idxs]
                        matched_in_region = np.argpartition(np.abs(region_counts-near_mean_transcript_count),num_near)[:num_near]
                        far_idxs = region_idxs[matched_in_region]
                    else:
                        raise Exception ("'comparison' not recognized")
                    
                    # Update if everything is matched correctly
                    sub_prox_labels[near_bool] = "Near"
                    sub_prox_labels[far_idxs] = "Far"
                except:
                    pass
            
        prox_labels[mouse_bool] = sub_prox_labels
    
    return (prox_labels)


def get_stats_df (adata, near_ages, cutoff_multiplier, celltype, ct, label=None,
                  min_pairs=50):
    '''
    Computes all statistics associated with proximity effect analysis including the proximity effect
    
        adata [AnnData] - object containing the spatial transcriptomics data
        near_ages [arr] - array with "Near", "Far", "Other" values
        cutoff_multiplier [float] - cutoff multiplier (only saved to the output)
        celltype [str] - effector cell type label (only saved to the output)
        ct [str] - target cell type label (only saved to the output)
        full_adata - unused argument [will remove in future iterations]
        label [str] - additional label to add to first column of output if not None
        min_pairs [int] - minimum number of cells in near/far set to compute proximity effect for
        
    Returns:
        df [Dataframe] - containing the following columns:
            "label", label if any
            "Near Cell", effector cell type name
            "AgeAccel Cell", target cell type name
            "n", cutoff multiplier used
            "t", t-test statistic
            "p", p-value from t-test
            "Aging Effect", Proximity Effect
            "Near Freq", normalized frequency of interactions
            "Near Num", number of interactions
    '''
    # run test
    t,p = ttest_ind(adata.obs['normalized_age_acceleration'].copy()[near_ages=="Near"],
                    adata.obs['normalized_age_acceleration'].copy()[near_ages=="Far"],
                    nan_policy='omit')
    
    # compute aging effect
    group1 = adata.obs['normalized_age_acceleration'].copy()[near_ages=="Near"]
    group2 = adata.obs['normalized_age_acceleration'].copy()[near_ages=="Far"]
    n1 = np.sum(~np.isnan(group1))
    n2 = np.sum(~np.isnan(group2))
    if (n1 > min_pairs) and (n2 > min_pairs):
        mean_diff = np.nanmean(group1)-np.nanmean(group2)
        pooled_sd = np.sqrt((n1*np.nanstd(group1)**2 + n2*np.nanstd(group2)**2) / (n1+n2)) 
        aging_effect = mean_diff/pooled_sd # cohen's d = mean_diff/pooled_sd
    else:
        aging_effect = np.nan
    
    # compute normalized near freq
    near_freq = np.sum(near_ages=="Near") / len(near_ages)
    near_num = np.sum(near_ages=="Near")
    
    # compile stats
    if label is None:
        df = pd.DataFrame([[celltype, ct, cutoff_multiplier, t, p, aging_effect, near_freq, near_num]],
                          columns=["Near Cell", "AgeAccel Cell", "n", "t", "p", "Aging Effect", "Near Freq", "Near Num"])
    else:
        df = pd.DataFrame([[label, celltype, ct, cutoff_multiplier, t, p, aging_effect, near_freq, near_num]],
                          columns=["label", "Near Cell", "AgeAccel Cell", "n", "t", "p", "Aging Effect", "Near Freq", "Near Num"])
    
    return (df)