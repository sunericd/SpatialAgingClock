import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, lasso_path, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
import pickle
import os


#### Normalization


def normalize_adata (adata, zscore=False):
    '''
    Helper function for normalizing gene expression
    '''

    # Normalize total to 250
    sc.pp.normalize_total(adata, target_sum=250)

    # Log transform
    sc.pp.log1p(adata)

    # Z-score (generally don't need to do)
    if zscore is True:
        sc.pp.scale(adata, max_value=10)
        
    return(adata)


#### Cross-validation (for building clocks)

def get_cv_iterator(adata, obs_name):
    '''
    Gets an sklearn-compatible CV iterator for unique values of adata.obs[obs_name]
    '''
    cv_iterator = []
    
    n = adata.shape[0]
    
    for i in np.unique(adata.obs[obs_name]):
        trainIndices = (np.arange(n)[adata.obs[obs_name]!=i]).astype(int)
        testIndices =  (np.arange(n)[adata.obs[obs_name]==i]).astype(int)
        cv_iterator.append( (trainIndices, testIndices) )
    
    return(cv_iterator)

    
#### SpatialSmooth


def spatial_smoothing_expression (adata, graph_method="fixed", adjacency_method="binary", group_obs="mouse_id",
                      n_neighbors=15, alpha=0.1, max_iter=100, tol=1e-3):
    '''
    Spatial smoothing for gene expression (i.e. SpatialSmooth)
    
    Refer to spatial_propagation.py for argument descriptions
    '''    
    # init objects for results
    pb_X = adata.X.copy()
    
    # run for each group_obs
    for g in np.unique(adata.obs[group_obs]):
        
        sub_adata = adata[adata.obs[group_obs]==g].copy()
        
        if sub_adata.shape[0] > 1:
            # build graph and adjacency matrix
            build_spatial_graph(sub_adata, method=graph_method, n_neighbors=n_neighbors)
            calc_adjacency_weights(sub_adata, method=adjacency_method)

            X = sub_adata.X.copy()
            S = sub_adata.obsp["S"]

            # propagate predictions
            smoothed_X = propagate (X, S, alpha, max_iter=max_iter, tol=tol, verbose=False)
                
            # append results
            pb_X[adata.obs[group_obs]==g, :] = smoothed_X
            
        else:
            pb_X[adata.obs[group_obs]==g, :] = sub_adata.X.copy()
        
    adata.X = pb_X
        
    return(adata)




def update (X, Xt, S, alpha):
    '''
    Update equation shared by reinforce() and smooth()
    '''
    Xt1 = (1-alpha)*X + alpha*(S@Xt)
    return(Xt1)
    
    
def propagate (X, S, alpha, max_iter=100, tol=1e-2, verbose=True):
    '''
    Iterate update() until convergence is reached. See reinforce() and smooth() for usage/argument details
        X is the numpy matrix of node values to propagate
        S is the adjacency matrix
        
    verbose = whether to print propagation iterations
    '''
    # independent updates
    Xt = X.copy()
    Xt1 = update(X, Xt, S, alpha)

    iter_num = 1
    while (iter_num < max_iter) and np.any(np.divide(np.abs(Xt1-Xt), np.abs(Xt), out=np.full(Xt.shape,0.0), where=Xt!=0) > tol):
        Xt = Xt1
        Xt1 = update(X, Xt, S, alpha)
        iter_num += 1
    
    if verbose is True:
        print("Propagation converged after "+str(iter_num)+" iterations")
    
    return(Xt1)


def calc_adjacency_weights (adata, method="cosine", beta=0.0):
    '''
    Creates a normalized adjacency matrix containing edges weights for spatial graph
        adata [AnnData] = spatial data, must include adata.obsp['spatial_connectivities'] and adata.obsp['spatial_distances']
        method [str] = "binary" (weight is binary - 1 if edge exists, 0 otherwise); "cluster" (one weight for same-cluster and different weight for diff-cluster neighbors); "cosine" (weight based on cosine similarity between neighbor gene expressions)
        beta [float] = only used when method is "cluster"; between 0 and 1; specifies the non-same-cluster edge weight relative to 1 (for same cluster edge weight)
    
    Adds adata.obsp["S"]:
        S [numpy matrix] = normalized weight adjacency matrix; nxn where n is number of cells in adata
    '''
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    # adjacency matrix from adata
    A = adata.obsp['spatial_connectivities']
    
    # compute weights
    if method == "binary":
        pass
    elif method == "cluster":
        # cluster AnnData if not already clustered
        if "cluster" not in adata.obs.columns:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=15)
            sc.tl.leiden(adata, key_added = "cluster")
        # init same and diff masks
        cluster_ids = adata.obs['cluster'].values
        same_mask = np.zeros(A.shape)
        for i in range(A.shape[1]):
            same_mask[:,i] = [1 if cid==cluster_ids[i] else 0 for cid in cluster_ids]
        diff_mask = np.abs(same_mask-1)
        # construct cluster-based adjacency matrix
        A = A*same_mask + A*diff_mask*beta
    elif method == "cosine":
        # PCA reduced space
        scaler = StandardScaler()
        pca = PCA(n_components=5, svd_solver='full')
        if isinstance(adata.X,np.ndarray):
            pcs = pca.fit_transform(scaler.fit_transform(adata.X))
        else:
            pcs = pca.fit_transform(scaler.fit_transform(adata.X.toarray()))
        # cosine similarities
        cos_sim = cosine_similarity(pcs)
        # update adjacency matrix
        A = A*cos_sim
        A[A < 0] = 0
    else:
        raise Exception ("weighting must be 'binary', 'cluster', 'cosine'")
    
    # normalized adjacency matrix
    S = normalize(A, norm='l1', axis=1)
    
    # update adata
    adata.obsp["S"] = S

def build_spatial_graph (adata, method="delaunay_radius", spatial="spatial", radius=None, n_neighbors=20, set_diag=True):
    '''
    Builds a spatial graph from AnnData according to specifications:
        adata [AnnData] - spatial data, must include adata.obsm[spatial]
        method [str]:
            - "radius" (all cells within radius are neighbors)
            - "delaunay" (triangulation)
            - "delaunay_radius" (triangulation with pruning by max radius; DEFAULT)
            - "fixed" (the k-nearest cells are neighbors determined by n_neighbors)
            - "fixed_radius" (knn by n_neighbors with pruning by max radius)
        spatial [str] - column name for adata.obsm to retrieve spatial coordinates
        radius [None or float/int] - radius around cell centers for which to detect neighbor cells; defaults to Q3+1.5*IQR of delaunay (or fixed for fixed_radius) neighbor distances
        n_neighbors [None or int] - number of neighbors to get for each cell (if method is "fixed" or "fixed_radius" or "radius_fixed"); defaults to 20
        set_diag [True or False] - whether to have diagonal of 1 in adjacency (before normalization); False is identical to theory and True is more robust; defaults to True
    
    Performs all computations inplace. Uses SquidPy implementations for graphs.
    '''
    if adata.shape[0] <= n_neighbors:
        n_neighbors=adata.shape[0]-1
    
    # delaunay graph
    if method == "delaunay": # triangulation only
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
    
    # radius-based methods
    elif method == "radius": # radius only
        if radius is None: # compute 90th percentile of delaunay triangulation
            sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", set_diag=set_diag)
    elif method == "delaunay_radius":
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0 # for computability
    elif method == "fixed_radius":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0 # for computability
            
    # fixed neighborhood size methods
    elif method == "fixed":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
            
    else:
        raise Exception ("method not recognized")
        
        
#### Pseudobulk (for non-spatial data)

def pseudobulk(adata, ident_cols, n=15, obs_to_average=None, obs_to_first=None,
               obsm_to_average=None, obsm_to_first=None, B=False, method="random", random_state=444):
    '''
    Inputs:
        adata [anndata]
            - AnnData object to pseudobulk where rows are single cells
        ident_col [list of str]
            - key in adata.obs where unique values specify pools from which to pseudobulk cells
        n [int]
            - number of single cells to sample from each unique pool to construct each pseudocell
        obs_to_average [None, str or list of str]
            - name of adata.obs columns containing quantities to average for each pseudocell
        obs_to_first [None, str or list of str]
            - name of adata.obsm column containing quantities to take first instance of for each pseudocell
        obsm_to_average [None, str or list of str]
            - name of adata.obs columns containing quantities to average for each pseudocell
        obsm_to_first [None, str or list of str]
            - name of adata.obsm column containing quantities to take first instance of for each pseudocell
        B [int or False]
            - number of total pseudocells created per unique identifier pool
            - can specify False to automatically use the size of the original pool
                - for method=="spatial", this will return pseudocell values in the same order as adata.X
        method [str]
            - "random" for random grouping for pseudobulking
            - "spatial" for spatially nearest neighbors grouping for pseudobulking
    
    Returns:
        pb_adata [anndata]
            - AnnData object where observations are pseudocells
    '''  
    # init groupings based on fused identifiers from ident_cols
    grouping_df = adata.obs.groupby(ident_cols).size().reset_index().rename(columns={0:'count'})
    
    # init objects for pseudobulk results
    pb_X = []
    pb_obs = {}
    pb_obsm = {}
    
    if obs_to_average is not None:
        if isinstance(obs_to_average, str):
            pb_obs[obs_to_average] = []
        else:
            for obs_name in obs_to_average:
                pb_obs[obs_name] = []
    
    if obs_to_first is not None:
        if isinstance(obs_to_first, str):
            pb_obs[obs_to_first] = []
        else:
            for obs_name in obs_to_first:
                pb_obs[obs_name] = []
                
    if obsm_to_average is not None:
        if isinstance(obsm_to_average, str):
            pb_obsm[obsm_to_average] = []
        else:
            for obs_name in obsm_to_average:
                pb_obsm[obs_name] = []
    
    if obsm_to_first is not None:
        if isinstance(obsm_to_first, str):
            pb_obsm[obsm_to_first] = []
        else:
            for obs_name in obsm_to_first:
                pb_obsm[obs_name] = []
    
    # subset into each unique pool and construct pseudocells
    for g in range(grouping_df.shape[0]):
        
        # iterative subsetting
        sub_adata = adata
        for idx in range(len(ident_cols)):
            sub_adata = sub_adata[sub_adata.obs[ident_cols[idx]] == grouping_df[ident_cols[idx]].values[g]]
        
        if sub_adata.shape[0] > 0: # pseudobulk if at least one cell in group
            
            # pseudobulking
            if B is False:
                B_ind = sub_adata.shape[0]
            else:
                B_ind = B

            # determine pseudobulking groups
            if method == "random":
                bootstrap_indices = []
                np.random.seed(random_state)
                random_seeds = np.random.randint(0,1e6,B_ind)
                for b in range(B_ind):
                    np.random.seed(random_seeds[b])
                    bootstrap_indices.append(np.random.choice(np.arange(sub_adata.shape[0]),n))

            elif method == "spatial":
                # compute nearest neighbors
                num_neigh = np.min([n+1, sub_adata.shape[0]-1])
                nbrs = NearestNeighbors(n_neighbors=num_neigh).fit(sub_adata.obsm["spatial"])
                distances_local, indices_local = nbrs.kneighbors(sub_adata.obsm["spatial"])
                # accumulate indices for each cell
                bootstrap_indices = []
                if B is False:
                    cell_center_idxs = np.arange(sub_adata.shape[0])
                else:
                    np.random.seed(random_state)
                    cell_center_idxs = np.random.choice(np.arange(sub_adata.shape[0]),B_ind)
                for cell_idx in cell_center_idxs:
                    bootstrap_indices.append([ni for ni in indices_local[cell_idx,1:num_neigh]])

            for bootstrap_idx in bootstrap_indices:
                # pseudocell expression (average)
                pb_X.append(sub_adata.X[bootstrap_idx,:].mean(axis=0)) # pseudocell

                # pseudocell metadata
                if obs_to_average is not None:
                    if isinstance(obs_to_average, str):
                        obs_to_average = [obs_to_average]
                    for obs_name in obs_to_average:
                        pb_obs[obs_name].append(sub_adata.obs[obs_name].iloc[bootstrap_idx].values.mean())

                if obs_to_first is not None:
                    if isinstance(obs_to_first, str):
                        obs_to_first = [obs_to_first]
                    for obs_name in obs_to_first:
                        pb_obs[obs_name].append(sub_adata.obs[obs_name].iloc[bootstrap_idx].values[0])

                if obsm_to_average is not None:
                    if isinstance(obsm_to_average, str):
                        obsm_to_average = [obsm_to_average]
                    for obs_name in obsm_to_average:
                        pb_obsm[obs_name].append(np.array(sub_adata.obsm[obs_name])[bootstrap_idx,:].mean(axis=0))

                if obsm_to_first is not None:
                    if isinstance(obsm_to_first, str):
                        obsm_to_first = [obsm_to_first]
                    for obs_name in obsm_to_first:
                        pb_obsm[obs_name].append(np.array(sub_adata.obsm[obs_name])[bootstrap_idx[0],:].copy())
                    
    # compile new AnnData object
    pb_X = np.vstack(pb_X)
    pb_adata = ad.AnnData(X=pb_X, dtype="float64")
    
    pb_meta_df = pd.DataFrame.from_dict(pb_obs)
    pb_meta_df.index = pb_adata.obs_names
    pb_adata.obs = pb_meta_df
    
    for key in pb_obsm:
        pb_adata.obsm[key] = np.vstack(pb_obsm[key])
    
    # include original information
    pb_adata.var=adata.var
    pb_adata.varm=adata.varm
    pb_adata.varp=adata.varp
    pb_adata.uns=adata.uns

    return(pb_adata)