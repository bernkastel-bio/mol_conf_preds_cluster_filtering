import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy import cluster
from sklearn.cluster import DBSCAN, HDBSCAN, AffinityPropagation
from scipy.stats import spearmanr
from astropy.stats.circstats import circmean, circstd

import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from typing import List, Dict, Optional, Union, Tuple, Any
from tqdm import tqdm
from icecream import ic


def circular_mean_rad(angles, period=2*np.pi):
    """
    Calculate circular mean for angular data in radians.
    
    Parameters:
    -----------
    angles : np.ndarray
        Array of angles in radians
    period : float
        Period of the circular data (default 2π)
    
    Returns:
    --------
    mean_angle : float
        Circular mean in radians
    """
    if len(angles) == 0:
        return 0.0
    
    # Normalize angles to [0, 2π] range
    normalized_angles = angles % period
    
    # Calculate mean using vector averaging
#     cos_sum = np.sum(np.cos(normalized_angles))
#     sin_sum = np.sum(np.sin(normalized_angles))
    
#     mean_angle = np.arctan2(sin_sum, cos_sum)
    
    mean_angle = circmean(normalized_angles)
    
    # Ensure result is in [0, 2π]
    if mean_angle < 0:
        mean_angle += period
    
    return mean_angle


def circular_std_rad(angles, period=2*np.pi):
    """
    Calculate circular standard deviation for angular data.
    
    Parameters:
    -----------
    angles : np.ndarray
        Array of angles in radians
    period : float
        Period of the circular data
    
    Returns:
    --------
    std : float
        Circular standard deviation
    """
    if len(angles) <= 1:
        return 0.0
    
    # Normalize angles to [0, 2π] range
    normalized_angles = angles % period
    
    # Calculate resultant vector length
#     cos_sum = np.sum(np.cos(normalized_angles))
#     sin_sum = np.sum(np.sin(normalized_angles))
    
#     R = np.sqrt(cos_sum**2 + sin_sum**2) / len(angles)
    
#     # Circular standard deviation
#     # std = sqrt(-2 * ln(R)) for data in radians
#     if R > 0:
#         std = np.sqrt(-2 * np.log(R))
#     else:
#         std = float('inf')  # Undefined for completely dispersed data
    std = circstd(normalized_angles, method='angular')
    
    return std


def circular_distance(
        angles1: np.ndarray, 
        angles2: np.ndarray, 
        period: float = 2*np.pi
        ):
    """
    Calculate circular distance between angles.
    
    Parameters:
    -----------
    angles1, angles2 : np.ndarray or float
        Angles to compare
    period : float
        Period of the circular data
    
    Returns:
    --------
    distance : np.ndarray or float
        Minimal circular distance
    """
    diff = np.abs((angles1 - angles2 + period / 2) % period - period / 2)
    return diff


def torsion_distance_matrix_batch(
    torsion_array: np.ndarray, 
    bond_periods: np.ndarray
    ) -> List[np.ndarray]:
    n_samples, n_bonds = torsion_array.shape
    distance_matrices = []
    # TODO:
    for bond_idx in range(n_bonds):
        dist_matrix = np.zeros((n_samples, n_samples))
        # distances = []
        period = bond_periods[bond_idx]
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = circular_distance(torsion_array[i, bond_idx], 
                                         torsion_array[j, bond_idx], 
                                         period)
                # distances.append(dist)
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist_matrix[i,j]
        distance_matrices.append(dist_matrix)
    
    return distance_matrices


def torsion_to_cartesian(
    torsion_angles_pred, 
    bond_periods
    ) -> Tuple[np.ndarray]:
    n_samples, n_bonds = torsion_angles_pred.shape
    
    # Project each bond to 2D circle, then apply PCA to reduce dimensions
    torsion_circle_coords = np.zeros((n_bonds, n_samples, 2))
    
    for bond_idx in range(n_bonds):
        period = bond_periods[bond_idx]
        normalized_angles = torsion_angles_pred[:, bond_idx] % period
        
        # Map to unit circle
        torsion_circle_coords[bond_idx, :, 0] = np.cos(normalized_angles)
        torsion_circle_coords[bond_idx, :, 1] = np.sin(normalized_angles)
    
    # Flatten for PCA but maintain bond structure awareness
    torsion_flat = torsion_circle_coords.reshape(n_samples, -1)
    return torsion_flat, torsion_circle_coords


def cluster_single_bond_angles(
        torsion_array: np.ndarray, 
        bond_periods: np.ndarray, 
        metric: str = 'precomputed',
        method: str = 'dbscan', 
        DBSCAN_params: Dict[str, Any] = {
            'eps': 0.3, 
            'min_samples': 3, 
            'metric_params': None, 
            'algorithm': 'auto', 
            'leaf_size': 30, 
            'p': None, 
            'n_jobs': -1
            },
        HDBSCAN_params: Dict[str, Any] = {
            'min_cluster_size': 5,
            'min_samples': None,
            'cluster_selection_epsilon': 0.7,
            'alpha': 1.0,
            'algorithm': 'auto',
            'cluster_selection_method': 'eom',
            'store_centers': None,
            'n_jobs': -1
            },
        AffinityProp_params: Dict[str, Any] = {
            'damping': 0.5, 
            'max_iter': 300, 
            'convergence_iter': 30, 
            'copy': True, 
            'preference': None, 
            },
        uid: str = 'Unknown'
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Cluster torsion angles for a single rotatable bond using circular-aware methods.
    
    Parameters:
    -----------
    bond_angles : np.ndarray, shape (n_samples,)
        Torsion angles for one bond across all samples
    bond_period : float
        Periodicity of the bond (usually 2π)
    method : str
        Clustering method ('affinity', 'dbscan', 'kmeans')
    
    Returns:
    --------
    cluster_labels : np.ndarray, shape (n_samples,)
        Cluster assignments
    cluster_centers : np.ndarray, shape (n_clusters,)
        Circular mean of each cluster
    """
    if metric == 'precomputed':
        distance_matrices = torsion_distance_matrix_batch(torsion_array, bond_periods)

        bond_clusters = []
        bond_cluster_centers = []
        # Clustering using precomputed distance matrices
        for bond_idx, dist_matrix in tqdm(enumerate(distance_matrices), total=len(distance_matrices), desc=f'Clustering angles for {uid} with precomputed angle distances via {method}'):
            
            if method.lower() == 'affinity':
                clustering = AffinityPropagation(**AffinityProp_params, affinity='precomputed').fit(dist_matrix)

                cluster_centers = []
                angles = torsion_array[:, bond_idx]
                labels = clustering.labels_
                
                for label in np.unique(labels):
                    cluster_centers.append(angles[clustering.cluster_centers_indices_[label]])

                bond_cluster_centers.append(np.array(cluster_centers))

            elif method.lower() == 'dbscan':
                clustering = DBSCAN(**DBSCAN_params, metric='precomputed').fit(dist_matrix)
            
            elif method.lower() == 'hdbscan':
                # When using a precomputed distance matrix, scikit-learn's HDBSCAN
                # does not allow storing centers. Ensure it is disabled.
                _hdbscan_params = {**HDBSCAN_params}
                _hdbscan_params['store_centers'] = None
                clustering = HDBSCAN(**_hdbscan_params, metric='precomputed').fit(dist_matrix)
            
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            labels = clustering.labels_
            bond_clusters.append(labels)
    
    elif metric == 'euclidean':
        torsion_flat, torsion_circle_coords = torsion_to_cartesian(torsion_array, bond_periods)
        
        bond_clusters = []
        bond_cluster_centers = []
        # cluster_centers_indices = []
        n_bonds = len(bond_periods)
        for bond_idx in tqdm(range(n_bonds), total=n_bonds, desc=f'Clustering angles for {uid} in cartesian coordinates via {method}'):
            bond_angles = torsion_circle_coords[bond_idx, :, :]
            
            if method.lower() == 'affinity':
                clustering = AffinityPropagation(**AffinityProp_params, affinity='euclidean').fit(bond_angles)
                
                cluster_centers = []
                angles = torsion_array[:, bond_idx]
                labels = clustering.labels_
                
                for label in np.unique(labels):
                    cluster_centers.append(angles[clustering.cluster_centers_indices_[label]])

                bond_cluster_centers.append(np.array(cluster_centers))
            
            elif method.lower() == 'dbscan':
                clustering = DBSCAN(**DBSCAN_params, metric='euclidean').fit(bond_angles)
            
            elif method.lower() == 'hdbscan':
                # Euclidean metric supports storing centers; use provided params
                clustering = HDBSCAN(**HDBSCAN_params, metric='euclidean').fit(bond_angles)
            
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            labels = clustering.labels_
            bond_clusters.append(labels)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if method.lower() == 'affinity':
        return bond_clusters, bond_cluster_centers
    
    else:
        bond_cluster_centers = []
        for i, angle_clusters in enumerate(bond_clusters):
            angles = torsion_array[:, i]
        
            # Calculate circular cluster centers
            unique_labels = np.unique(angle_clusters[angle_clusters != -1])  # Exclude noise
            cluster_centers = []
        
            for label in unique_labels:
                cluster_angles = angles[angle_clusters == label]
                # Circular mean calculation
                center_cos = np.mean(np.cos(cluster_angles))
                center_sin = np.mean(np.sin(cluster_angles))
                center_angle = np.arctan2(center_sin, center_cos)
                cluster_centers.append(center_angle)
            
            bond_cluster_centers.append(np.array(cluster_centers))
        
        return bond_clusters, bond_cluster_centers


def analyze_all_bonds_separately(
        torsion_angles_pred: np.ndarray, 
        torsion_angles_true: np.ndarray, 
        bond_periods: np.ndarray, 
        scores: Optional[np.ndarray] = None, 
        uid: str = "Unknown", 
        max_bonds_to_plot: int = 10,
        clustering_method: str = 'hdbscan',
        clustering_metric: str = 'precomputed',
        print_stats: bool = False,
        plot_first_n: int = 5,
        fig_savepath: Optional[str] = None,
        complex_idx: int = None,
        DBSCAN_params: Dict[str, Any] = {
            'eps': 0.3, 
            'min_samples': 3, 
            'metric_params': None, 
            'algorithm': 'auto', 
            'leaf_size': 30, 
            'p': None, 
            'n_jobs': -1
            },
        HDBSCAN_params: Dict[str, Any] = {
            'min_cluster_size': 5,
            'min_samples': None,
            'cluster_selection_epsilon': 0.7,
            'alpha': 1.0,
            'algorithm': 'auto',
            'cluster_selection_method': 'eom',
            'store_centers': None,
            'n_jobs': -1
            },
        AffinityProp_params: Dict[str, Any] = {
            'damping': 0.9, 
            'max_iter': 1000, 
            'convergence_iter': 100, 
            'copy': True, 
            'preference': None, 
            },
        ) -> Dict[str, Any]:
    """
    Perform per-bond clustering analysis for all rotatable bonds.
    
    Parameters:
    -----------
    torsion_angles_pred : np.ndarray, shape (n_samples, n_bonds)
        Predicted torsion angles
    torsion_angles_true : np.ndarray, shape (n_bonds,)
        True torsion angles
    bond_periods : np.ndarray, shape (n_bonds,)
        Periodicity for each bond
    scores : np.ndarray, shape (n_samples,), optional
        Confidence scores
    uid : str
        Complex identifier for titles
    max_bonds_to_plot : int
        Maximum number of bonds to plot (for performance)
    
    Returns:
    --------
    bond_analyses : dict
        Analysis results for each bond
    """
    
    n_samples, n_bonds = torsion_angles_pred.shape
    bond_analyses = {}
    
    print(f"Analyzing {n_bonds} rotatable bond(s) for {uid}")
    
    # Limit plotting for performance with many bonds
    bonds_to_plot = min(n_bonds, max_bonds_to_plot)

    # Perform clustering
    all_bonds_cluster_labels, all_bonds_cluster_centers = cluster_single_bond_angles(
        torsion_array=torsion_angles_pred, 
        bond_periods=bond_periods, 
        metric=clustering_metric, 
        method=clustering_method,
        uid=uid,
        DBSCAN_params=DBSCAN_params,
        HDBSCAN_params=HDBSCAN_params,
        AffinityProp_params=AffinityProp_params
        )

    figs = []
    
    for bond_idx in range(n_bonds):
        bond_angles = torsion_angles_pred[:, bond_idx]
        true_angle = torsion_angles_true[bond_idx]
        period = bond_periods[bond_idx]
        
        # print(f"Bond {bond_idx}: {n_samples} samples, period: {period:.3f} rad")
        
        cluster_labels = all_bonds_cluster_labels[bond_idx]
        cluster_centers = all_bonds_cluster_centers[bond_idx]

        # Calculate bond-specific statistics
        bond_stats = calculate_bond_statistics(
            torsion_angles=bond_angles, 
            true_angle=true_angle, 
            bond_period=period, 
            cluster_labels=cluster_labels, 
            scores=scores
            )
        
        # Store analysis
        bond_analyses[bond_idx] = {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'statistics': bond_stats,
            'n_clusters': len(cluster_centers)
            }
        
        # Generate visualization (only for first few bonds to avoid too many plots)
        if complex_idx < plot_first_n:
            # if bond_idx <= bonds_to_plot:
            fig_single_bond = plot_single_bond_clustering(
                bond_angles=bond_angles, 
                true_angle=true_angle, 
                period=period, 
                cluster_labels=cluster_labels, 
                cluster_centers=cluster_centers, 
                bond_stats=bond_stats, 
                bond_idx=bond_idx, 
                scores=scores, 
                uid=uid,
                print_stats=print_stats
                )

            plt.show()
            figs.append(fig_single_bond)
            if fig_savepath is not None:
                fig_single_bond.savefig(os.path.join(fig_savepath, f"{uid}_bond_{bond_idx}_tor_clusters.png"), dpi=300)

        # elif bond_idx > bonds_to_plot:
        #     print(f"(Skipping visualization for bond {bond_idx} to limit plot count)")
    
    return figs, bond_analyses


def plot_single_bond_clustering(
        bond_angles: np.ndarray, 
        true_angle: float, 
        period: float, 
        cluster_labels: np.ndarray, 
        cluster_centers: np.ndarray, 
        bond_stats: Dict[str, Any], 
        bond_idx: int, 
        scores: np.ndarray = None, 
        uid: str = "Unknown",
        print_stats: bool = False
        ) -> plt.Figure:
    """
    Create comprehensive visualization for a single bond's clustering results.
    """
    
    n_samples = len(bond_angles)
    n_clusters = len(cluster_centers)
    bond_angles = bond_angles % period
    
    # Create figure with multiple subplots
    # fig = plt.figure(figsize=(20, 18))
    fig, axes = plt.subplots(2, 2, figsize=(22, 20), layout='constrained')
    fig.suptitle(f"Bond {bond_idx} Analysis - {uid}\n"
                f"True: {true_angle:.3f} rad, {n_clusters} clusters, "
                f"Convergence: {bond_stats['convergence_quality']:.3f}", 
                fontsize=22) # , y=1.05
    
    # 1. Circular scatter plot with clustering
    # ax1 = fig.add_subplot(141)
    ax1 = axes[0, 0]
    plot_circular_clustering(
        ax=ax1, 
        angles=bond_angles, 
        true_angle=true_angle, 
        cluster_labels=cluster_labels, 
        cluster_centers=cluster_centers, 
        title="Circular Clustering"
        )
    
    # 2. Linear histogram with clustering colors
    # ax2 = fig.add_subplot(142)
    ax2 = axes[0, 1]
    plot_clustered_histogram(
        ax=ax2, 
        angles=bond_angles, 
        true_angle=true_angle, 
        cluster_labels=cluster_labels, 
        period=period, 
        title="Clustered Distribution"
        )
    
    # 3. Angle vs Score scatter (if scores available)
    # ax3 = fig.add_subplot(143)
    ax3 = axes[1, 0]
    if scores is not None:
        plot_angle_vs_score(
            ax=ax3, 
            angles=bond_angles,
            scores=scores, 
            cluster_labels=cluster_labels, 
            true_angle=true_angle,
            title="Angles vs Confidence Scores"
            )
    else:
        plot_angle_timeline(
            ax=ax3, 
            angles=bond_angles, 
            true_angle=true_angle, 
            title="Angle Distribution"
            )
    
    # 4. Cluster statistics
    # ax4 = fig.add_subplot(144)
    ax4 = axes[1, 1]
    plot_cluster_statistics(
        ax=ax4, 
        angles=bond_angles, 
        scores=scores,
        cluster_labels=cluster_labels, 
        cluster_centers=cluster_centers,
        bond_stats=bond_stats, 
        title="Cluster Analysis"
        )
    
    # plt.tight_layout()
    # plt.close()
    
    # Print detailed statistics
    if print_stats:
        print_single_bond_statistics(bond_stats, bond_idx)

    return fig


def plot_circular_clustering(
        ax: Axes, 
        angles: np.ndarray, 
        true_angle: float, 
        cluster_labels: np.ndarray, 
        cluster_centers: np.ndarray, 
        title: str
        ) -> Axes:
    """Circular scatter plot showing clustering results"""
    
    # Convert to Cartesian for plotting
    # cos_vals = np.cos(angles)
    # sin_vals = np.sin(angles)
    # true_cos = np.cos(true_angle)
    # true_sin = np.sin(true_angle)
    
    # Normalize angles to [-pi, pi] before plotting
    angles_normalized = ((angles + np.pi) % (2 * np.pi)) - np.pi
    true_angle_normalized = ((true_angle + np.pi) % (2 * np.pi)) - np.pi
    
    # Convert to Cartesian for plotting
    cos_vals = np.cos(angles_normalized)
    sin_vals = np.sin(angles_normalized)
    true_cos = np.cos(true_angle_normalized)
    true_sin = np.sin(true_angle_normalized)
    
    # Plot clusters with different colors
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # Noise points (DBSCAN)
            mask = cluster_labels == label
            ax.scatter(cos_vals[mask], sin_vals[mask], color='gray', alpha=0.3, 
                      s=60, label='Noise')
        else:
            mask = cluster_labels == label
            ax.scatter(cos_vals[mask], sin_vals[mask], color=colors[i], alpha=0.7, 
                      s=90, label=f'Cluster {label}')
    
    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        center_cos = np.cos(center)
        center_sin = np.sin(center)
        ax.scatter(center_cos, center_sin, color=colors[i], s=120, marker='h', 
                  edgecolors='black', linewidth=3)
    
    # Plot true angle
    ax.scatter(true_cos, true_sin, color='red', s=200, marker='*', 
              edgecolors='black', label='True', zorder=5)
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.3)
    ax.add_patch(circle)
    
    ax.set_xlabel('cos(θ)', fontsize=18)
    ax.set_ylabel('sin(θ)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_clustered_histogram(
        ax: Axes, 
        angles: np.ndarray, 
        true_angle: float, 
        cluster_labels: np.ndarray, 
        period: float, 
        title: str
        ) -> Axes:
    """Histogram colored by cluster membership"""
    
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # # Plot histogram bars by cluster
    # bottom = np.zeros(len(angles))
    # bin_edges = np.linspace(-period/2, period/2, 30)
    
    # Center angles around -pi to pi for proper periodic visualization
    angles_normalized = ((angles + np.pi) % (2 * np.pi)) - np.pi
    true_angle_normalized = ((true_angle + np.pi) % (2 * np.pi)) - np.pi
    
    # Plot histogram bars by cluster, initialize bottom for stacking
    bin_edges = np.linspace(-np.pi, np.pi, 30)  # Fixed range for clear visualization
    bottom = np.zeros(len(bin_edges) - 1)  # Initialize bottom for stacked bars
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        # cluster_angles = angles[mask]
        cluster_angles = angles_normalized[mask]
        
        if len(cluster_angles) > 0:
            counts, bins = np.histogram(cluster_angles, bins=bin_edges)
            # ax.bar(bins[:-1], counts, width=bins[1]-bins[0], alpha=0.7, 
            #        color=colors[i], label=f'Cluster {label}', bottom=bottom[:len(counts)])
            ax.bar(bins[:-1], counts, width=bins[1]-bins[0], alpha=0.7, 
                   color=colors[i], label=f'Cluster {label}', bottom=bottom)
            # Update bottom for stacked effect (optional)
            # bottom += counts
    
    # Plot true angle line
    ax.axvline(true_angle, color='red', linestyle='--', linewidth=3, label='True')
    
    ax.set_xlabel('Angle (radians)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    return ax


def plot_angle_vs_score(
        ax: Axes, 
        angles: np.ndarray, 
        scores: np.ndarray, 
        cluster_labels: np.ndarray, 
        true_angle: float, 
        title: str
        ) -> Axes:
    """Scatter plot of angles vs confidence scores, colored by cluster"""
    # Normalize angles to [-pi, pi] for consistent visualization
    angles_normalized = ((angles + np.pi) % (2 * np.pi)) - np.pi
    true_angle_normalized = ((true_angle + np.pi) % (2 * np.pi)) - np.pi
    
    # Set x-axis limits explicitly
    ax.set_xlim(-np.pi, np.pi)
    
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = cluster_labels == label
            # ax.scatter(angles[mask], scores[mask], color='gray', alpha=0.4, 
            ax.scatter(angles_normalized[mask], scores[mask], color='gray', alpha=0.4, 
                       s=60, label='Noise')
        else:
            mask = cluster_labels == label
            # ax.scatter(angles[mask], scores[mask], color=colors[i], alpha=0.64, 
            ax.scatter(angles_normalized[mask], scores[mask], color=colors[i], alpha=0.64, 
                      s=90, label=f'Cluster {label}')
    
    # # Mark true angle
    # ax.axvline(true_angle, color='red', linestyle='--', linewidth=3, label='True')
    
    # Mark true angle using normalized value
    ax.axvline(true_angle_normalized, color='red', linestyle='--', linewidth=3, label='True')
    
    ax.set_xlabel('Angle (radians)', fontsize=18)
    ax.set_ylabel('Confidence Score', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    return ax


def plot_angle_timeline(
        ax: Axes, 
        angles: np.ndarray, 
        true_angle: float, 
        title: str
        ) -> Axes:
    """
    Plot angles as a timeline or distribution.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    angles : np.ndarray
        Array of angles
    true_angle : float
        True angle value
    title : str
        Plot title
    """
    n_samples = len(angles)
    
    # Create a simple distribution plot
    ax.hist(angles, bins=min(30, n_samples//2), alpha=0.7, color='skyblue', density=True)
    ax.axvline(true_angle, color='red', linestyle='--', linewidth=3, label=f'True: {true_angle:.3f}')
    
    # Add some statistics
    mean_angle = circular_mean_rad(angles)
    std_angle = circular_std_rad(angles)
    
    ax.axvline(mean_angle, color='blue', linestyle='-', linewidth=1, alpha=0.7, 
               label=f'Mean: {mean_angle:.3f}')
    
    ax.set_xlabel('Angle (radians)', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    return ax


def calculate_bond_statistics(
        torsion_angles: np.ndarray, 
        true_angle: float, 
        bond_period: float, 
        cluster_labels: np.ndarray, 
        scores: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
    """Calculate comprehensive statistics for a single bond"""
    
    n_samples = len(torsion_angles)
    unique_labels = np.unique(cluster_labels, return_counts=False)
    
    # Basic circular statistics
    circular_mean = circular_mean_rad(torsion_angles, bond_period)
    circular_std = circular_std_rad(torsion_angles, bond_period)
    # errors = circular_distance(torsion_angles, true_angle, bond_period)
    # Use absolute errors for proper minimization
    errors = circular_distance(torsion_angles, true_angle, bond_period)
    # errors = np.abs(raw_errors)
    
    # Cluster statistics
    cluster_stats = {}
    best_cluster_label = -1
    best_cluster_size_label = -1
    best_cluster_score_label = -1
    best_cluster_score = 1000
    best_cluster_size = 1
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_angles = torsion_angles[mask]
        cluster_size = np.sum(mask)
        # cluster_errors = errors[mask]
        # cluster_raw_errors = raw_errors[mask]
        cluster_errors = errors[mask]
        
        if cluster_size > 0:
            cluster_mean = circular_mean_rad(cluster_angles, bond_period)
            cluster_std = circular_std_rad(cluster_angles, bond_period)
            # mean_cluster_error = np.mean(cluster_raw_errors)
            # median_cluster_error = np.median(cluster_raw_errors)
            mean_cluster_error = np.mean(cluster_errors)
            median_cluster_error = np.median(cluster_errors)
            
            if best_cluster_size < cluster_size:
                best_cluster_size = cluster_size
                best_cluster_size_label = label
            
            if scores is not None:
                mean_cluster_score = np.mean(scores[mask])
                median_cluster_score = np.median(scores[mask])
                if median_cluster_score < best_cluster_score:
                    best_cluster_score = median_cluster_score
                    best_cluster_score_label = label
            else:
                mean_cluster_score = None
                median_cluster_score = None
            
            cluster_stats[label] = {
                'size': cluster_size,
                'mean_angle': cluster_mean,
                'std': cluster_std,
                'mean_error': mean_cluster_error,
                'median_error': median_cluster_error,
                'mean_score': mean_cluster_score,
                'median_score': median_cluster_score,
                'min_score_angle_idx': np.argmin(scores[mask]) if scores is not None else None,
                'min_score_angles_global_idx': cluster_indices[np.argmin(scores[mask])] if scores is not None else None,
                'min_score_angle': cluster_angles[np.argmin(scores[mask])] if scores is not None else None,
                'min_score_angle_error': cluster_errors[np.argmin(scores[mask])] if scores is not None else None,
                'min_score': scores[mask][np.argmin(scores[mask])] if scores is not None else None,
                'min_error_angle_idx': np.argmin(cluster_errors),
                'min_error_angles_global_idx': cluster_indices[np.argmin(cluster_errors)],
                'min_error_angle': cluster_angles[np.argmin(cluster_errors)],
                'min_error': cluster_errors[np.argmin(cluster_errors)],
                'min_error_angle_score': scores[mask][np.argmin(cluster_errors)] if scores is not None else None
                }
    
    if best_cluster_score_label == -1:
        if best_cluster_size_label == -1:
            best_cluster_label = -1
        else:
            best_cluster_label = best_cluster_size_label
    
    else:
        if best_cluster_size_label == -1:
            best_cluster_label = best_cluster_score_label
        else:
            if best_cluster_score_label == best_cluster_size_label:
                best_cluster_label = best_cluster_score_label
            else:
                best_cluster_label = best_cluster_size_label
    
    # Overall statistics
    stats = {
        'n_samples': n_samples,
        'true_angle': true_angle,
        'circular_mean': circular_mean,
        'circular_std': circular_std,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'total_min_error': np.min(errors),  # Already using absolute errors
        'total_min_error_angle_idx': np.argmin(errors),  # Using absolute errors
        'total_min_error_angle': torsion_angles[np.argmin(errors)],  # Using absolute errors
        'mean_score': np.mean(scores) if scores is not None else None,
        'median_score': np.median(scores) if scores is not None else None,
        'total_min_score': np.min(scores) if scores is not None else None,
        'total_min_score_angle_idx': np.argmin(scores) if scores is not None else None,
        'total_min_score_angle': torsion_angles[np.argmin(scores)] if scores is not None else None,
        'best_cluster_label': best_cluster_label,
        'convergence_quality': circular_std / bond_period,  # Lower = more converged
        'n_clusters': len(unique_labels),
        'n_noise': len(cluster_labels[cluster_labels == -1]),
        'cluster_stats': cluster_stats,
        'largest_cluster_size': max([s['size'] for s in cluster_stats.values()]) if cluster_stats else 0,
        'largest_cluster_frac': max([s['size'] for s in cluster_stats.values()]) / n_samples if cluster_stats else 0
        }
    
    if scores is not None:
        stats['score_error_correlation'] = spearmanr(errors, scores).correlation
    
    return stats


def plot_cluster_statistics(
        ax: Axes, 
        angles: np.ndarray, 
        cluster_labels: np.ndarray, 
        cluster_centers: np.ndarray,
        bond_stats: Dict[str, Any],
        title: str,
        scores: Optional[np.ndarray] = None
        ) -> Axes:
    """
    Plot statistical summary of clustering results.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    angles : np.ndarray
        Array of angles
    cluster_labels : np.ndarray
        Cluster assignments
    cluster_centers : np.ndarray
        Cluster center angles
    bond_stats : dict
        Bond statistics dictionary
    title : str
        Plot title
    """
    # Remove spines and ticks for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Prepare text content
    stats_text = []
    
    # Basic statistics
    stats_text.append(f"Samples: {bond_stats['n_samples']}")
    stats_text.append(f"True angle: {bond_stats['true_angle']:.3f} rad")
    stats_text.append(f"Circular mean: {bond_stats['circular_mean']:.3f} rad")
    stats_text.append(f"Circular std: {bond_stats['circular_std']:.3f} rad")
    stats_text.append(f"Median error: {bond_stats['median_error']:.3f} rad")
    if bond_stats['median_score'] is not None:
        stats_text.append(f"Median score: {bond_stats['median_score']:.3f}")
    else:
        stats_text.append(f"Median score: -")
    stats_text.append(f"Min error: {bond_stats['total_min_error']:.3f} rad")
    if bond_stats['total_min_score'] is not None:
        stats_text.append(f"Min score: {bond_stats['total_min_score']:.3f}")
    else:
        stats_text.append(f"Min score: -")
    stats_text.append(f"Min errror angle index: {bond_stats['total_min_error_angle_idx']}")
    if bond_stats['total_min_score_angle_idx'] is not None:
        stats_text.append(f"Min score angle index: {bond_stats['total_min_score_angle_idx']}")
    else:
        stats_text.append(f"Min score angle index: -")
    stats_text.append(f"Convergence: {bond_stats['convergence_quality']:.3f}")
    stats_text.append(f"Clusters: {bond_stats['n_clusters']}")
    stats_text.append(f"Noise: {bond_stats['n_noise']}")
    stats_text.append(f"Best cluster: {bond_stats['best_cluster_label']}")
    stats_text.append(f"Largest cluster: {bond_stats['largest_cluster_frac']:.1%}")
    
    if 'score_error_correlation' in bond_stats:
        stats_text.append(f"Score corr: {bond_stats['score_error_correlation']:.3f}")
    
    # Cluster details
    stats_text.append("\nBest cluster Details:")
    # for cluster_id, cluster_info in bond_stats['cluster_stats'].items():
    cluster_id = bond_stats['best_cluster_label']
    cluster_info = bond_stats['cluster_stats'][cluster_id]
    # stats_text.append(f"Cluster {cluster_id}:")
    stats_text.append(f"Size: {cluster_info['size']}")
    stats_text.append(f"Mean angle: {cluster_info['mean_angle']:.3f} rad")
    stats_text.append(f"Mean error: {cluster_info['mean_error']:.3f} rad")
    stats_text.append(f"Median error: {cluster_info['median_error']:.3f} rad")
    if cluster_info['median_score'] is not None:
        stats_text.append(f"Median Score: {cluster_info['median_score']:.3f}")
    else:
        stats_text.append(f"Median Score: -")
    stats_text.append(f"Min error: {cluster_info['min_error']:.3f}")
    stats_text.append(f"Min error angle index: {cluster_info['min_error_angle_idx']}")
    stats_text.append(f"Min error angle: {cluster_info['min_error_angle']:.3f} rad")
    if cluster_info['min_error_angle_score'] is not None:
        stats_text.append(f"Min error angle score: {cluster_info['min_error_angle_score']:.3f}")
    else:
        stats_text.append(f"Min error angle score: -")
    if cluster_info['min_score'] is not None:
        stats_text.append(f"Min score: {cluster_info['min_score']:.3f}")
    else:
        stats_text.append(f"Min score: -")
    if cluster_info['min_score_angle_idx'] is not None:
        stats_text.append(f"Min score angle index: {cluster_info['min_score_angle_idx']}")
    else:
        stats_text.append(f"Min score angle index: -")
    if cluster_info['min_score_angle'] is not None:
        stats_text.append(f"Min score angle: {cluster_info['min_score_angle']:.3f}")
    else:
        stats_text.append(f"Min score angle: -")
    if cluster_info['min_score_angle_error'] is not None:
        stats_text.append(f"Min score angle error: {cluster_info['min_score_angle_error']:.3f} rad")
    else:
        stats_text.append(f"Min score angle error: -")
    
    # Display text
    ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
            fontfamily='monospace', fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    return ax


def print_single_bond_statistics(
        bond_stats: Dict[str, Any],
        bond_idx: int
        ) -> None:
    """Print detailed statistics for a single bond"""
    
    print(f"Bond {bond_idx} Statistics:")
    print(f"Samples: {bond_stats['n_samples']}")
    print(f"True angle: {bond_stats['true_angle']:.3f} rad")
    print(f"Circular mean: {bond_stats['circular_mean']:.3f} rad")
    print(f"Circular std: {bond_stats['circular_std']:.3f} rad")
    print(f"Mean error: {bond_stats['mean_error']:.3f} rad")
    print(f"Convergence: {bond_stats['convergence_quality']:.3f}")
    print(f"Clusters: {bond_stats['n_clusters']}")
    print(f"Largest cluster: {bond_stats['largest_cluster_frac']:.1%}")

    if 'score_error_correlation' in bond_stats:
        print(f"Score-error correlation: {bond_stats['score_error_correlation']:.3f}")

    print("Cluster details:")
    for cluster_id, cluster_info in bond_stats['cluster_stats'].items():
        print(f"Cluster {cluster_id}: {cluster_info['size']} samples, "
              f"mean angle: {cluster_info['mean_angle']:.3f}, "
              f"error: {cluster_info['mean_error']:.3f}")
    print()


def calculate_aggregate_torsion_metrics(
        bond_analyses: Dict[str, Any]
        ):
    """
    Calculate aggregate metrics across all bonds
    Get bond_analyses as an output of 'analyze_all_bonds_separately'
    """
    
    all_mean_errors = []
    all_convergence = []
    all_n_clusters = []
    
    for bond_idx, analysis in bond_analyses.items():
        stats = analysis['statistics']
        all_mean_errors.append(stats['mean_error'])
        all_convergence.append(stats['convergence_quality'])
        all_n_clusters.append(stats['n_clusters'])
    
    return {
        'mean_error_across_bonds': np.mean(all_mean_errors),
        'median_error_across_bonds': np.median(all_mean_errors),
        'mean_convergence': np.mean(all_convergence),
        'mean_clusters_per_bond': np.mean(all_n_clusters),
        'worst_converged_bond': np.argmax(all_convergence) if all_convergence else None
        }