import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN, HDBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from typing import List, Dict, Optional, Union, Tuple, Any
from tqdm import tqdm
from icecream import ic

import utils


def compute_conformation_rmsd_matrix(
        conformations: np.ndarray, 
        isomorphisms: Optional[List] = None,
        use_symmetry: bool = True,
        max_timeout: int = 2
        ) -> np.ndarray:
    """
    Compute pairwise RMSD distance matrix between molecular conformations.
    
    Parameters:
    -----------
    conformations : np.ndarray, shape (n_conformations, n_atoms, 3)
        Array of molecular conformations
    isomorphisms : list, optional
        Graph isomorphisms for symmetry correction
    use_symmetry : bool
        Whether to use symmetry-corrected RMSD
    max_timeout : float
        Maximum time per RMSD calculation (seconds)
    
    Returns:
    --------
    distance_matrix : np.ndarray, shape (n_conformations, n_conformations)
        Pairwise RMSD distance matrix
    """
    n_conformations = conformations.shape[0]
    distance_matrix = np.zeros((n_conformations, n_conformations))
    
    print(f"Computing RMSD distance matrix for {n_conformations} conformations...")
    
    for i in tqdm(range(n_conformations), desc="Computing RMSD distances"):
        for j in range(i + 1, n_conformations):
            pos_a = conformations[i]
            pos_b = conformations[j]
            
            # Compute rigid alignment
            rot, tr = utils.find_rigid_alignment(pos_a, pos_b)
            pos_a_aligned = (pos_a - pos_a.mean(0).reshape(1, 3)) @ rot.T + tr.reshape(1, 3)
            
            if use_symmetry and isomorphisms is not None:
                try:
                    with utils.time_limit(max_timeout):
                        rmsd = utils.get_symmetry_rmsd_with_isomorphisms(
                            pos_a_aligned, 
                            pos_b, 
                            isomorphisms, 
                            max_timeout
                            )

                except utils.TimeoutException:
                    # Fallback to regular RMSD if symmetry calculation times out
                    rmsd = np.sqrt(((pos_a_aligned - pos_b) ** 2).sum(axis=1).sum() / pos_a.shape[0])
            else:
                rmsd = np.sqrt(((pos_a_aligned - pos_b) ** 2).sum(axis=1).sum() / pos_a.shape[0])
            
            distance_matrix[i, j] = rmsd
            distance_matrix[j, i] = rmsd  # Symmetric matrix
    
    return distance_matrix


def cluster_conformations(
        conformations: np.ndarray,
        scores: np.ndarray,
        true_conformation: Optional[np.ndarray] = None,
        isomorphisms: Optional[List] = None,
        method: str = 'hdbscan',
        use_symmetry: bool = True,
        symm_rmsd_max_timeout: float = 2,
        DBSCAN_params: Dict[str, Any] = {
            'eps': 2.0,
            'min_samples': 3,
            'metric': 'precomputed'
            },
        HDBSCAN_params: Dict[str, Any] = {
            'min_cluster_size': 3,
            'min_samples': None,
            'cluster_selection_epsilon': 0.0,
            'alpha': 1.0,
            'metric': 'precomputed',
            'cluster_selection_method': 'eom',
            'store_centers': None,
            'n_jobs': -1
            },
        AffinityProp_params: Dict[str, Any] = {
            'damping': 0.5,
            'max_iter': 200,
            'convergence_iter': 15,
            'affinity': 'precomputed',
            'preference': None
            }
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Cluster molecular conformations based on RMSD distances.
    
    Parameters:
    -----------
    conformations : np.ndarray, shape (n_conformations, n_atoms, 3)
        Molecular conformations to cluster
    scores : np.ndarray, shape (n_conformations,)
        Confidence scores (lower is better)
    true_conformation : np.ndarray, optional, shape (n_atoms, 3)
        True conformation for validation
    isomorphisms : list, optional
        Graph isomorphisms for symmetry correction
    method : str
        Clustering method ('dbscan', 'hdbscan', 'affinity')
    use_symmetry : bool
        Whether to use symmetry-corrected RMSD
    DBSCAN_params, HDBSCAN_params, AffinityProp_params : dict
        Parameters for respective clustering algorithms
    
    Returns:
    --------
    cluster_labels : np.ndarray
        Cluster assignments for each conformation
    distance_matrix : np.ndarray
        Pairwise RMSD distance matrix
    cluster_info : dict
        Detailed cluster analysis information
    """
    # Compute distance matrix
    distance_matrix = compute_conformation_rmsd_matrix(
        conformations, 
        isomorphisms, 
        use_symmetry,
        symm_rmsd_max_timeout
        )
    
    # Perform clustering
    if method.lower() == 'dbscan':
        clustering = DBSCAN(**DBSCAN_params).fit(distance_matrix)
    elif method.lower() == 'hdbscan':
        clustering = HDBSCAN(**HDBSCAN_params).fit(distance_matrix)
    elif method.lower() == 'affinity':
        clustering = AffinityPropagation(**AffinityProp_params).fit(distance_matrix)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    cluster_labels = clustering.labels_
    
    # Analyze clusters
    cluster_info = analyze_conformation_clusters(
        conformations=conformations, 
        cluster_labels=cluster_labels, 
        scores=scores, 
        distance_matrix=distance_matrix, 
        isomorphisms=isomorphisms,
        true_conformation=true_conformation,
        symm_rmsd_max_timeout=symm_rmsd_max_timeout
        )
    
    return cluster_labels, distance_matrix, cluster_info


def analyze_conformation_clusters(
        conformations: np.ndarray,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        distance_matrix: np.ndarray,
        isomorphisms: Optional[List] = None,
        true_conformation: Optional[np.ndarray] = None,
        symm_rmsd_max_timeout: float = 2
        ) -> Dict[str, Any]:
    """
    Analyze conformation clusters and compute quality metrics.
    
    Parameters:
    -----------
    conformations : np.ndarray
        Molecular conformations
    cluster_labels : np.ndarray
        Cluster assignments
    scores : np.ndarray
        Confidence scores
    distance_matrix : np.ndarray
        Pairwise RMSD distance matrix
    true_conformation : np.ndarray, optional
        True conformation for validation
    
    Returns:
    --------
    cluster_info : dict
        Comprehensive cluster analysis
    """
    n_conformations = len(conformations)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise
    
    cluster_info = {
        'n_conformations': n_conformations,
        'n_clusters': n_clusters,
        'n_noise': np.sum(cluster_labels == -1),
        'silhouette_score': None,
        'cluster_stats': {},
        'best_cluster': None,
        'overall_stats': {}
        }
    
    # Compute silhouette score if we have multiple clusters
    if n_clusters > 1:
        try:
            distance_silhouette_matrix = np.fill_diagonal(distance_matrix, 0)
            cluster_info['silhouette_score'] = silhouette_score(
                distance_silhouette_matrix, cluster_labels, metric='precomputed'
                )
        except:
            cluster_info['silhouette_score'] = None
    
    # Analyze each cluster
    best_cluster_score = float('inf')
    best_cluster_id = None
    
    for label in unique_labels:
        if label == -1:  # Noise points
            mask = cluster_labels == label
            cluster_scores = scores[mask]
            cluster_confs = conformations[mask]
            
            cluster_info['cluster_stats'][label] = {
                'size': np.sum(mask),
                'median_score': np.median(cluster_scores) if len(cluster_scores) > 0 else np.inf,
                'mean_score': np.mean(cluster_scores) if len(cluster_scores) > 0 else np.inf,
                'min_score': np.min(cluster_scores) if len(cluster_scores) > 0 else np.inf,
                'is_noise': True
                }
        else:
            mask = cluster_labels == label
            cluster_indices = np.where(mask)[0]
            cluster_scores = scores[mask]
            min_score_id = np.argmin(cluster_scores)
            global_min_score_id = cluster_indices[np.argmin(cluster_scores)]
            cluster_confs = conformations[mask]
            cluster_distances = distance_matrix[np.ix_(mask, mask)]
            
            # Compute intra-cluster statistics
            intra_cluster_distances = cluster_distances[np.triu_indices_from(cluster_distances, k=1)]
            cluster_compactness = np.mean(intra_cluster_distances) if len(intra_cluster_distances) > 0 else 0
            
            # Find cluster centroid (conformation closest to cluster mean)
            cluster_mean_distances = np.mean(cluster_distances, axis=1)
            centroid_idx = np.argmin(cluster_mean_distances)
            cluster_centroid = cluster_confs[centroid_idx]
            
            cluster_info['cluster_stats'][label] = {
                'size': np.sum(mask),
                'median_score': np.median(cluster_scores),
                'mean_score': np.mean(cluster_scores),
                'min_score': np.min(cluster_scores),
                'min_score_idx': min_score_id,
                'global_min_score_idx': global_min_score_id,
                'compactness': cluster_compactness,
                'centroid_idx': centroid_idx,
                'centroid_score': cluster_scores[centroid_idx],
                'is_noise': False
                }
            
            # Track best cluster (lowest median score)
            if np.median(cluster_scores) < best_cluster_score:
                best_cluster_score = np.median(cluster_scores)
                best_cluster_id = label
    
    cluster_info['best_cluster'] = best_cluster_id
    
    # Overall statistics
    cluster_info['overall_stats'] = {
        'mean_score': np.mean(scores),
        'median_score': np.median(scores),
        'min_score': np.min(scores),
        'score_std': np.std(scores),
        'best_conformation_idx': np.argmin(scores),
        'best_conformation_score': np.min(scores)
        }
    
    # Compute distances to true conformation if available
    if true_conformation is not None:
        true_distances = []
        for i, conf in enumerate(conformations):
            rot, tr = utils.find_rigid_alignment(conf, true_conformation)
            conf_aligned = (conf - conf.mean(0).reshape(1, 3)) @ rot.T + tr.reshape(1, 3)
            try:
                with utils.time_limit(symm_rmsd_max_timeout):
                    rmsd = utils.get_symmetry_rmsd_with_isomorphisms(
                        conf_aligned, 
                        true_conformation, 
                        isomorphisms, 
                        symm_rmsd_max_timeout
                        )

            except utils.TimeoutException:
                # Fallback to regular RMSD if symmetry calculation times out
                rmsd = np.sqrt(((conf_aligned - true_conformation) ** 2).sum(axis=1).sum() / conf.shape[0])
            true_distances.append(rmsd)
        
        cluster_info['overall_stats']['true_rmsd_mean'] = np.mean(true_distances)
        cluster_info['overall_stats']['true_rmsd_median'] = np.median(true_distances)
        cluster_info['overall_stats']['true_rmsd_min'] = np.min(true_distances)
        cluster_info['overall_stats']['true_rmsd_min_idx'] = np.argmin(true_distances)
        
        # Add true RMSD to cluster stats
        for label in unique_labels:
            if label != -1:
                mask = cluster_labels == label
                cluster_true_rmsds = [true_distances[i] for i in np.where(mask)[0]]
                cluster_min_rmsd_id = np.where(mask)[0][np.argmin(cluster_true_rmsds)]
                min_score_rmsd = cluster_true_rmsds[cluster_info['cluster_stats'][label]['min_score_idx']]
                cluster_info['cluster_stats'][label]['true_rmsd_mean'] = np.mean(cluster_true_rmsds)
                cluster_info['cluster_stats'][label]['true_rmsd_median'] = np.median(cluster_true_rmsds)
                cluster_info['cluster_stats'][label]['true_rmsd_min'] = np.min(cluster_true_rmsds)
                cluster_info['cluster_stats'][label]['true_rmsd_min_idx'] = cluster_min_rmsd_id
                cluster_info['cluster_stats'][label]['true_rmsd_min_score'] = min_score_rmsd
    
    return cluster_info


def select_best_cluster_conformations(
        conformations: np.ndarray,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        cluster_info: Dict[str, Any],
        selection_method: str = 'best_centroid',
        n_conformations: int = 1
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Select the best conformations from the best cluster.
    
    Parameters:
    -----------
    conformations : np.ndarray
        All conformations
    cluster_labels : np.ndarray
        Cluster assignments
    scores : np.ndarray
        Confidence scores
    cluster_info : dict
        Cluster analysis information
    selection_method : str
        Method to select conformations ('best_centroid', 'best_scores', 'representative')
    n_conformations : int
        Number of conformations to select
    
    Returns:
    --------
    selected_conformations : np.ndarray
        Selected conformations
    selected_indices : np.ndarray
        Indices of selected conformations
    selection_info : dict
        Information about the selection process
    """
    best_cluster_id = cluster_info['best_cluster']
    
    if best_cluster_id is None:
        # No clusters found, select globally best conformations
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:n_conformations]
        selected_conformations = conformations[selected_indices]
        
        selection_info = {
            'method': 'global_best',
            'cluster_id': None,
            'selection_reason': 'No clusters found, selected globally best conformations'
            }
        
        return selected_conformations, selected_indices, selection_info
    
    # Get conformations from best cluster
    cluster_mask = cluster_labels == best_cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_conformations = conformations[cluster_mask]
    cluster_scores = scores[cluster_mask]
    
    if selection_method == 'best_centroid':
        # Select centroid and best scoring conformations
        centroid_idx = cluster_info['cluster_stats'][best_cluster_id]['centroid_idx']
        centroid_global_idx = cluster_indices[centroid_idx]
        
        # Get remaining best scores
        remaining_scores = cluster_scores[np.arange(len(cluster_scores)) != centroid_idx]
        remaining_indices = cluster_indices[np.arange(len(cluster_indices)) != centroid_idx]
        
        sorted_remaining = np.argsort(remaining_scores)
        selected_indices = [centroid_global_idx] + list(remaining_indices[sorted_remaining[:n_conformations-1]])
        
    elif selection_method == 'best_scores':
        # Select best scoring conformations from cluster
        sorted_cluster_scores = np.argsort(cluster_scores)
        selected_local_indices = sorted_cluster_scores[:n_conformations]
        selected_indices = cluster_indices[selected_local_indices]
        
    elif selection_method == 'representative':
        # Select diverse conformations from cluster
        if len(cluster_conformations) <= n_conformations:
            selected_indices = cluster_indices
        else:
            # Select centroid and diverse conformations
            centroid_idx = cluster_info['cluster_stats'][best_cluster_id]['centroid_idx']
            centroid_global_idx = cluster_indices[centroid_idx]
            
            # Simple diversity selection (could be improved with more sophisticated methods)
            remaining_indices = cluster_indices[np.arange(len(cluster_indices)) != centroid_idx]
            selected_indices = [centroid_global_idx] + list(remaining_indices[:n_conformations-1])
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    selected_conformations = conformations[selected_indices]
    
    selection_info = {
        'method': selection_method,
        'cluster_id': best_cluster_id,
        'cluster_size': len(cluster_indices),
        'selected_scores': scores[selected_indices],
        'selection_reason': f'Selected from best cluster {best_cluster_id} using {selection_method}'
        }
    
    return selected_conformations, selected_indices, selection_info


def plot_conformation_clustering(
        conformations: np.ndarray,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        distance_matrix: np.ndarray,
        cluster_info: Dict[str, Any],
        true_conformation: Optional[np.ndarray] = None,
        uid: str = "Unknown",
        figsize: Tuple[int, int] = (22, 30)
        ) -> plt.Figure:
    """
    Create comprehensive visualization of conformation clustering results.
    
    Parameters:
    -----------
    conformations : np.ndarray
        Molecular conformations
    cluster_labels : np.ndarray
        Cluster assignments
    scores : np.ndarray
        Confidence scores
    distance_matrix : np.ndarray
        Pairwise RMSD distance matrix
    cluster_info : dict
        Cluster analysis information
    true_conformation : np.ndarray, optional
        True conformation
    uid : str
        Complex identifier
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    # gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distance matrix heatmap
    # ax1 = fig.add_subplot(gs[0, 0])
    # plot_distance_matrix_heatmap(ax1, distance_matrix, cluster_labels, title="RMSD Distance Matrix")
    
    # 2. Cluster size distribution
    # ax2 = fig.add_subplot(gs[0, 1])
    # plot_cluster_size_distribution(ax2, cluster_labels, cluster_info, title="Cluster Sizes")
    
    # 3. Score vs cluster
    # ax3 = fig.add_subplot(gs[0, 2])
    # plot_score_vs_cluster(ax3, cluster_labels, scores, cluster_info, title="Scores by Cluster")
    
    # 4. PCA projection
    # ax4 = fig.add_subplot(gs[1, :2])
    # plot_conformation_pca(ax4, conformations, cluster_labels, scores, true_conformation, title="PCA Projection")
    
    # 5. Cluster statistics
    # ax5 = fig.add_subplot(gs[1, 2])
    # plot_cluster_statistics(ax5, cluster_info, title="Cluster Analysis")
    
    # 6. t-SNE projection
    # ax6 = fig.add_subplot(gs[2, :])
    # plot_conformation_tsne(ax6, conformations, cluster_labels, scores, true_conformation, title="t-SNE Projection")
    
    # New layout using nested GridSpec
    # Height ratios: row0 : row1 : row2 = 1 : 2 : 2.5 (row2 is 1.25x row1)
    outer_gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1, 2, 2.5],
        hspace=0.3
    )

    # Row 0: three equal-width subplots
    row0_gs = outer_gs[0].subgridspec(nrows=1, ncols=3, wspace=0.3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(row0_gs[0, 0])
    ax2 = fig.add_subplot(row0_gs[0, 1])
    ax3 = fig.add_subplot(row0_gs[0, 2])
    plot_distance_matrix_heatmap(ax1, distance_matrix, cluster_labels, title="RMSD Distance Matrix")
    plot_cluster_size_distribution(ax2, cluster_labels, cluster_info, title="Cluster Sizes")
    plot_score_vs_cluster(ax3, cluster_labels, scores, cluster_info, title="Scores by Cluster")

    # Row 1: PCA (2.5) : Stats (0.5)
    row1_gs = outer_gs[1].subgridspec(nrows=1, ncols=2, wspace=0.5, width_ratios=[2.3, 0.7])
    ax4 = fig.add_subplot(row1_gs[0, 0])
    ax5 = fig.add_subplot(row1_gs[0, 1])
    plot_conformation_pca(ax4, conformations, cluster_labels, scores, true_conformation, title="PCA Projection")
    plot_cluster_statistics(ax5, cluster_info, title="Cluster Analysis")

    # Row 2: t-SNE full width
    row2_gs = outer_gs[2].subgridspec(nrows=1, ncols=1)
    ax6 = fig.add_subplot(row2_gs[0, 0])
    plot_conformation_tsne(ax6, conformations, cluster_labels, scores, true_conformation, title="t-SNE Projection")
    
    if cluster_info['silhouette_score'] is not None:
        fig.suptitle(f"Conformation Clustering Analysis - {uid}\n"
                    f"Clusters: {cluster_info['n_clusters']}, "
                    f"Silhouette: {cluster_info['silhouette_score']:.3f}", 
                    fontsize=22, fontweight='bold')
    else:
        fig.suptitle(f"Conformation Clustering Analysis - {uid}\n"
                    f"Clusters: {cluster_info['n_clusters']}, "
                    f"Silhouette: Indefined for a single cluster", 
                    fontsize=22, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def plot_distance_matrix_heatmap(
        ax: Axes,
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        title: str
        ) -> Axes:
    """Plot RMSD distance matrix as heatmap with cluster annotations."""
    
    # Reorder by clusters for better visualization
    unique_labels = np.unique(cluster_labels)
    reordered_indices = []
    
    for label in unique_labels:
        if label == -1:
            # Put noise points at the end
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        reordered_indices.extend(cluster_indices)
    
    # Add noise points at the end
    noise_indices = np.where(cluster_labels == -1)[0]
    reordered_indices.extend(noise_indices)
    
    # Reorder distance matrix
    reordered_matrix = distance_matrix[np.ix_(reordered_indices, reordered_indices)]
    reordered_labels = cluster_labels[reordered_indices]
    
    # Create heatmap
    im = ax.imshow(reordered_matrix, cmap='viridis', aspect='auto')
    
    # Add cluster boundaries
    current_pos = 0
    for label in unique_labels:
        if label == -1:
            continue
        cluster_size = np.sum(reordered_labels == label)
        if cluster_size > 0:
            ax.axvline(current_pos, color='red', linewidth=2)
            ax.axhline(current_pos, color='red', linewidth=2)
            current_pos += cluster_size
    
    ax.set_xlabel('Conformation Index', fontsize=18)
    ax.set_ylabel('Conformation Index', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, )
    cbar.set_label('RMSD (Å)', fontsize=16)
    cbar.ax.tick_params(labelsize=14) 
    
    return ax


def plot_cluster_size_distribution(
        ax: Axes,
        cluster_labels: np.ndarray,
        cluster_info: Dict[str, Any],
        title: str
        ) -> Axes:
    """Plot distribution of cluster sizes."""
    
    cluster_sizes = []
    cluster_ids = []
    
    for label, stats in cluster_info['cluster_stats'].items():
        if not stats['is_noise']:
            cluster_sizes.append(stats['size'])
            cluster_ids.append(f'C{label}')
    
    if cluster_sizes:
        bars = ax.bar(cluster_ids, cluster_sizes, alpha=0.7, color='#97d700', edgecolor='black')
        ax.set_xlabel('Cluster ID', fontsize=18)
        ax.set_ylabel('Size', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(title, fontsize=20, fontweight='bold')
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(size), ha='center', va='bottom', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No clusters found', ha='center', va='center', fontsize=18, transform=ax.transAxes)
        ax.set_title(title)
    
    return ax


def plot_score_vs_cluster(
        ax: Axes,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        cluster_info: Dict[str, Any],
        title: str
        ) -> Axes:
    """Plot score distribution by cluster."""
    
    unique_labels = np.unique(cluster_labels)
    cluster_data = []
    cluster_names = []
    
    for label in unique_labels:
        if label == -1:
            cluster_names.append('Noise')
        else:
            cluster_names.append(f'C{label}')
        
        mask = cluster_labels == label
        cluster_scores = scores[mask]
        cluster_data.append(cluster_scores)
    
    # Create box plot
    bp = ax.boxplot(cluster_data, labels=cluster_names, patch_artist=True)
    
    # Color boxes by cluster quality
    for i, (patch, label) in enumerate(zip(bp['boxes'], unique_labels)):
        if label == -1:
            patch.set_facecolor('lightgray')
        elif label == cluster_info['best_cluster']:
            patch.set_facecolor('#97d700')
        else:
            patch.set_facecolor('#46b2ff')
    
    ax.set_xlabel('Cluster', fontsize=18)
    ax.set_ylabel('Score (lower is better)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    return ax


def plot_conformation_pca(
        ax: Axes,
        conformations: np.ndarray,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        true_conformation: Optional[np.ndarray] = None,
        title: str = "PCA Projection"
        ) -> Axes:
    """Plot conformations in PCA space."""
    
    # Flatten conformations for PCA
    n_conformations, n_atoms, n_coords = conformations.shape
    conformations_flat = conformations.reshape(n_conformations, -1)
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(conformations_flat)
    
    # Plot clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if label == -1:
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                      color='gray', alpha=0.7, s=50, label='Noise')
        else:
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                      color=colors[i], alpha=0.8, s=80, label=f'Cluster {label}')
    
    # Plot true conformation if available
    if true_conformation is not None:
        true_flat = true_conformation.reshape(1, -1)
        true_pca = pca.transform(true_flat)[0]
        ax.scatter(true_pca[0], true_pca[1], color='red', s=200, marker='*', 
                  edgecolors='black', label='True', zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=18)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_conformation_tsne(
        ax: Axes,
        conformations: np.ndarray,
        cluster_labels: np.ndarray,
        scores: np.ndarray,
        true_conformation: Optional[np.ndarray] = None,
        title: str = "t-SNE Projection"
        ) -> Axes:
    """Plot conformations in t-SNE space."""
    
    # Flatten conformations for t-SNE
    n_conformations, n_atoms, n_coords = conformations.shape
    conformations_flat = conformations.reshape(n_conformations, -1)
    
    # Add true conformation if available
    if true_conformation is not None:
        true_flat = true_conformation.reshape(1, -1)
        combined_data = np.vstack([conformations_flat, true_flat])
    else:
        combined_data = conformations_flat
    
    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_conformations-1))
    tsne_result = tsne.fit_transform(combined_data)
    
    # Separate results
    if true_conformation is not None:
        conformations_tsne = tsne_result[:-1]
        true_tsne = tsne_result[-1]
    else:
        conformations_tsne = tsne_result
    
    # Plot clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if label == -1:
            ax.scatter(conformations_tsne[mask, 0], conformations_tsne[mask, 1], 
                      color='gray', alpha=0.7, s=50, label='Noise')
        else:
            ax.scatter(conformations_tsne[mask, 0], conformations_tsne[mask, 1], 
                      color=colors[i], alpha=0.8, s=80, label=f'Cluster {label}')
    
    # Plot true conformation if available
    if true_conformation is not None:
        ax.scatter(true_tsne[0], true_tsne[1], color='red', s=200, marker='*', 
                  edgecolors='black', label='True', zorder=5)
    
    ax.set_xlabel('t-SNE 1', fontsize=18)
    ax.set_ylabel('t-SNE 2', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_cluster_statistics(
        ax: Axes,
        cluster_info: Dict[str, Any],
        title: str
        ) -> Axes:
    """Plot cluster statistics as text."""
    
    # Remove spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Prepare statistics text
    stats_text = []
    
    # Overall statistics
    overall = cluster_info['overall_stats']
    stats_text.append(f"Total Conformations: {cluster_info['n_conformations']}")
    stats_text.append(f"Number of Clusters: {cluster_info['n_clusters']}")
    stats_text.append(f"Noise Points: {cluster_info['n_noise']}")
    
    if cluster_info['silhouette_score'] is not None:
        stats_text.append(f"Silhouette Score: {cluster_info['silhouette_score']:.3f}")
    
    stats_text.append(f"Best Score: {overall['min_score']:.3f}")
    stats_text.append(f"Mean Score: {overall['mean_score']:.3f}")
    
    if 'true_rmsd_min' in overall:
        stats_text.append(f"Best True RMSD: {overall['true_rmsd_min']:.3f}Å")
        stats_text.append(f"Mean True RMSD: {overall['true_rmsd_mean']:.3f}Å")
    
    # Best cluster information
    if cluster_info['best_cluster'] is not None:
        best_cluster = cluster_info['best_cluster']
        best_stats = cluster_info['cluster_stats'][best_cluster]
        stats_text.append(f"\nBest Cluster: {best_cluster}")
        stats_text.append(f"Size: {best_stats['size']}")
        stats_text.append(f"Median Score: {best_stats['median_score']:.3f}")
        stats_text.append(f"Compactness: {best_stats['compactness']:.3f}Å")
        
        if 'true_rmsd_min' in best_stats:
            stats_text.append(f"Best True RMSD: {best_stats['true_rmsd_min']:.3f}Å")
    
    # Display text
    ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
            fontfamily='monospace', fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    return ax


def print_clustering_summary(
        cluster_info: Dict[str, Any],
        selection_info: Dict[str, Any],
        uid: str = "Unknown"
        ) -> None:
    """Print comprehensive clustering summary."""
    
    print(f"\n{'='*60}")
    print(f"CONFORMATION CLUSTERING SUMMARY - {uid}")
    print(f"{'='*60}")
    
    # Overall statistics
    overall = cluster_info['overall_stats']
    print(f"Total Conformations: {cluster_info['n_conformations']}")
    print(f"Number of Clusters: {cluster_info['n_clusters']}")
    print(f"Noise Points: {cluster_info['n_noise']}")
    
    if cluster_info['silhouette_score'] is not None:
        print(f"Silhouette Score: {cluster_info['silhouette_score']:.3f}")
    
    print(f"Score Range: {overall['min_score']:.3f} - {overall['mean_score'] + 2*overall['score_std']:.3f}")
    print(f"Mean Score: {overall['mean_score']:.3f} ± {overall['score_std']:.3f}")
    
    if 'true_rmsd_min' in overall:
        print(f"True RMSD Range: {overall['true_rmsd_min']:.3f} - {overall['true_rmsd_mean'] + 2*np.std([stats.get('true_rmsd_mean', 0) for stats in cluster_info['cluster_stats'].values()]):.3f}Å")
    
    # Cluster details
    print(f"\nCluster Details:")
    print(f"{'Cluster':<10} {'Size':<8} {'Med Score':<12} {'Compact':<10} {'True RMSD':<12}")
    print(f"{'-'*60}")
    
    for label, stats in cluster_info['cluster_stats'].items():
        if stats['is_noise']:
            print(f"{'Noise':<10} {stats['size']:<8} {stats['median_score']:<12.3f} {'N/A':<10} {'N/A':<12}")
        else:
            true_rmsd = stats.get('true_rmsd_min', 'N/A')
            if isinstance(true_rmsd, float):
                true_rmsd = f"{true_rmsd:.3f}"
            print(f"{label:<10} {stats['size']:<8} {stats['median_score']:<12.3f} {stats['compactness']:<10.3f} {true_rmsd:<12}")
    
    # Selection information
    print(f"\nSelection Results:")
    print(f"Method: {selection_info['method']}")
    if selection_info['cluster_id'] is not None:
        print(f"Selected from Cluster: {selection_info['cluster_id']}")
        print(f"Cluster Size: {selection_info['cluster_size']}")
    print(f"Selected Scores: {selection_info['selected_scores']}")
    print(f"Reason: {selection_info['selection_reason']}")
    
    print(f"{'='*60}\n")


def comprehensive_conformation_analysis(
        uid_data: Dict[str, Any],
        supplement_data: Dict[str, Any],
        uid: str,
        clustering_method: str = 'hdbscan',
        selection_method: str = 'best_centroid',
        n_conformations: int = 1,
        use_symmetry: bool = True,
        symm_rmsd_max_timeout: float = 2,
        plot_first_n: int = 20,
        fig_savepath: Optional[str] = None,
        complex_idx: int = None,
        print_summary: bool = True
        ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Comprehensive conformation clustering analysis for a single complex.
    
    Parameters:
    -----------
    uid_data : dict
        Data for the complex containing sample_metrics and true_pos
    supplement_data : dict
        Supplementary data containing isomorphisms
    uid : str
        Complex identifier
    clustering_method : str
        Clustering algorithm to use
    selection_method : str
        Method to select best conformations
    n_conformations : int
        Number of conformations to select
    use_symmetry : bool
        Whether to use symmetry-corrected RMSD
    plot_first_n : int
        Number of plots to produce
    print_summary : bool
        Whether to print summary
    
    Returns:
    --------
    clustering_results : dict
        Clustering analysis results
    selected_conformations : dict
        Selected conformations and metadata
    figures : dict
        Generated figures (if plot_results=True)
    """
    
    # Extract data
    sample_metrics = uid_data['sample_metrics']
    true_pos = uid_data['true_pos']
    
    conformations = np.array([sample['pred_pos'] for sample in sample_metrics])
    scores = np.array([sample['error_estimate_0'] for sample in sample_metrics])
    
    # Get isomorphisms if available
    isomorphisms = supplement_data.get(uid, {}).get('isomorphism', None)
    
    # Perform clustering
    cluster_labels, distance_matrix, cluster_info = cluster_conformations(
        conformations=conformations,
        scores=scores,
        true_conformation=true_pos,
        isomorphisms=isomorphisms,
        method=clustering_method,
        use_symmetry=use_symmetry,
        symm_rmsd_max_timeout=symm_rmsd_max_timeout
        )
    
    # Select best conformations
    selected_confs, selected_indices, selection_info = select_best_cluster_conformations(
        conformations=conformations,
        cluster_labels=cluster_labels,
        scores=scores,
        cluster_info=cluster_info,
        selection_method=selection_method,
        n_conformations=n_conformations
        )
    
    # Prepare results
    clustering_results = {
        'cluster_labels': cluster_labels,
        'distance_matrix': distance_matrix,
        'cluster_info': cluster_info,
        'clustering_method': clustering_method,
        'use_symmetry': use_symmetry
        }
    
    selected_conformations = {
        'conformations': selected_confs,
        'indices': selected_indices,
        'selection_info': selection_info,
        'scores': scores[selected_indices],
        'cluster_assignments': cluster_labels[selected_indices]
        }
    
    figures = {}
    
    # Generate plots if requested
    if complex_idx <= plot_first_n:
        fig = plot_conformation_clustering(
            conformations=conformations,
            cluster_labels=cluster_labels,
            scores=scores,
            distance_matrix=distance_matrix,
            cluster_info=cluster_info,
            true_conformation=true_pos,
            uid=uid
            )
        plt.show()
        figures['main_analysis'] = fig
        if fig_savepath:
            fig.savefig(os.path.join(fig_savepath, f"{uid}_conf_clusters.png"), dpi=300)
    
    # Print summary if requested
    if print_summary:
        print_clustering_summary(cluster_info, selection_info, uid)
    
    return clustering_results, selected_conformations, figures
