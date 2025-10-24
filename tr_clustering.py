import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN, AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
from icecream import ic



def cluster_translations(
        tr_preds: np.ndarray,
        scores: np.ndarray,
        method: str = 'dbscan',
        dbscan_params: Dict[str, Any] = {
            'eps': 0.7,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'leaf_size': 20,
            'n_jobs': -1
            },
        hdbscan_params: Dict[str, Any] = {
            'min_cluster_size': 5,
            'min_samples': None,
            'cluster_selection_epsilon': 0.0,
            'max_cluster_size': None,
            'metric': 'euclidean',
            'alpha': 1.0,
            'cluster_selection_method': 'eom',
            'store_centers': 'medoid',
            'algorithm': 'auto',
            'leaf_size': 20,
            'n_jobs': -1
            },
        affinity_params: Dict[str, Any] = {
            'damping': 0.9,
            'max_iter': 1000,
            'convergence_iter': 100,
            'affinity': 'euclidean',
            'random_state': 0xC0FFEE
            },
        auto_eps: bool = False
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster translation predictions using various algorithms.
    
    Parameters:
    -----------
    tr_preds : np.ndarray, shape (n_samples, 3)
        Translation predictions
    scores : np.ndarray, shape (n_samples,)
        Confidence scores (lower is better)
    method : str
        Clustering method: 'dbscan', 'hdbscan', 'affinity'
    dbscan_params : dict, optional
        Parameters for DBSCAN
    hdbscan_params : dict, optional
        Parameters for HDBSCAN
    affinity_params : dict, optional
        Parameters for AffinityPropagation
    auto_eps : bool
        Whether to automatically estimate eps for DBSCAN
        
    Returns:
    --------
    labels : np.ndarray
        Cluster labels (-1 for noise)
    cluster_info : dict
        Information about the clustering
    """
    n_samples = len(tr_preds)
    
    if n_samples == 0:
        return np.array([], dtype=int), {}
    
    # Auto-estimate eps for DBSCAN if requested
    if auto_eps and method == 'dbscan':
        eps = _estimate_eps_from_knn(tr_preds, k=min(5, n_samples))
        dbscan_params = dbscan_params.copy()
        dbscan_params['eps'] = eps
    
    # Perform clustering
    if method == 'dbscan':
        dbscan_params['min_samples'] = max(3, min(5, n_samples // 4))
        clustering = DBSCAN(**dbscan_params)
    elif method == 'hdbscan':
        hdbscan_params['min_cluster_size'] = max(3, min(5, n_samples // 4))
        clustering = HDBSCAN(**hdbscan_params)
    elif method == 'affinity':
        clustering = AffinityPropagation(**affinity_params)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    labels = clustering.fit_predict(tr_preds)
    
    # Calculate cluster statistics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1)
    
    cluster_info = {
        'method': method,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'unique_labels': unique_labels
        # 'clustering_params': {
        #     'dbscan': dbscan_params,
        #     'hdbscan': hdbscan_params,
        #     'affinity': affinity_params
        #     }[method]
        }
    
    return labels, cluster_info


def _estimate_eps_from_knn(
    tr_preds: np.ndarray, 
    k: int = 5
    ) -> float:
    """Estimate DBSCAN eps parameter from k-nearest neighbors distances."""
    if len(tr_preds) <= k:
        return 0.5
    
    nn = NearestNeighbors(n_neighbors=min(k, len(tr_preds)))
    nn.fit(tr_preds)
    distances, _ = nn.kneighbors(tr_preds)
    
    # Use the k-th nearest neighbor distance
    kth_distances = distances[:, -1]
    eps = float(np.percentile(kth_distances, 70))  # Use 70th percentile
    
    return max(eps, 0.1)  # Ensure minimum eps


def calculate_cluster_statistics(
        tr_preds: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        tr_true: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for each cluster.
    
    Parameters:
    -----------
    tr_preds : np.ndarray, shape (n_samples, 3)
        Translation predictions
    scores : np.ndarray, shape (n_samples,)
        Confidence scores
    labels : np.ndarray, shape (n_samples,)
        Cluster labels
    tr_true : np.ndarray, shape (3,), optional
        True translation for error calculation
        
    Returns:
    --------
    stats : dict
        Comprehensive cluster statistics
    """
    stats = {
        'clusters': {},
        'best_cluster': None,
        'total_best_point': None,
        'overall_stats': {}
        }
    
    unique_labels = np.unique(labels)
    cluster_medians = {}
    
    # Calculate statistics for each cluster
    for label in unique_labels:
        cluster_mask = labels == label
        
        if label == -1:  # Noise points
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{label}'
        
        cluster_tr = tr_preds[cluster_mask]
        cluster_scores = scores[cluster_mask]
        
        cluster_stats = {
            'size': np.sum(cluster_mask),
            'center_mean': np.mean(cluster_tr, axis=0),
            'center_median': np.median(cluster_tr, axis=0),
            'score_mean': np.mean(cluster_scores),
            'score_median': np.median(cluster_scores),
            'score_std': np.std(cluster_scores),
            'score_min': np.min(cluster_scores),
            'score_max': np.max(cluster_scores),
            'spread': np.mean(np.std(cluster_tr, axis=0)),
            'indices': np.where(cluster_mask)[0]
            }
        
        # Calculate distance to true if available
        if tr_true is not None:
            cluster_stats['center_error_mean'] = np.linalg.norm(cluster_stats['center_mean'] - tr_true)
            cluster_stats['center_error_median'] = np.linalg.norm(cluster_stats['center_median'] - tr_true)
            
            # Best point in cluster (lowest score)
            best_idx = np.argmin(cluster_scores)
            best_tr = cluster_tr[best_idx]
            cluster_stats['best_point'] = {
                'translation': best_tr,
                'score': cluster_scores[best_idx],
                'global_index': cluster_stats['indices'][best_idx],
                'error': np.linalg.norm(best_tr - tr_true)
                }
        
        stats['clusters'][cluster_name] = cluster_stats
        cluster_medians[label] = cluster_stats['score_median']
    
    # Find best cluster (lowest median score, excluding noise)
    valid_labels = unique_labels[unique_labels != -1]
    if len(valid_labels) > 0:
        best_cluster_label = valid_labels[np.argmin([cluster_medians[label] for label in valid_labels])]
        stats['best_cluster'] = {
            'label': best_cluster_label,
            'name': f'cluster_{best_cluster_label}',
            'stats': stats['clusters'][f'cluster_{best_cluster_label}']
            }
        # stats['best_cluster_label'] = best_cluster_label
        
        # Find overall best point
        if tr_true is not None:
            best_score = np.inf
            best_idx = None
            for cluster_name, cluster_data in stats['clusters'].items():
                if cluster_data['best_point'] is not None:
                    if cluster_data['best_point']['score'] < best_score:
                        best_score = cluster_data['best_point']['score']
                        best_idx = cluster_data['best_point']['global_index']
                        if cluster_name != 'noise':
                            cluster_label_w_best_point = cluster_name.split('_')[1]
                        else:
                            cluster_label_w_best_point = -1
            
            if best_idx is not None:
                stats['total_best_point'] = {
                    'cluster_label': cluster_label_w_best_point,
                    'translation': tr_preds[best_idx],
                    'score': scores[best_idx],
                    'index': best_idx,
                    'error': np.linalg.norm(tr_preds[best_idx] - tr_true) if tr_true is not None else None
                    }
    
    # Overall statistics
    stats['overall_stats'] = {
        'n_samples': len(tr_preds),
        'n_clusters': len(valid_labels),
        'n_noise': np.sum(labels == -1),
        'score_mean': np.mean(scores),
        'score_median': np.median(scores),
        'score_std': np.std(scores)
        }
    
    return stats


def select_best_cluster(
        stats: Dict[str, Any],
        selection_method: str = 'best_score'
        ) -> Tuple[int, Dict[str, Any]]:
    """
    Select the best cluster based on different criteria.
    
    Parameters:
    -----------
    stats : dict
        Cluster statistics from calculate_cluster_statistics
    selection_method : str
        Selection method: 'best_score', 'largest', 'most_compact'
        
    Returns:
    --------
    selected_label : int
        Label of the selected cluster
    selected_stats : dict
        Statistics of the selected cluster
    """
    if stats['best_cluster'] is None:
        return -1, {}
    
    clusters = stats['clusters']
    
    if selection_method == 'best_score':
        # Already calculated in stats
        selected_label = stats['best_cluster']['label']
        selected_stats = stats['best_cluster']['stats']
    
    elif selection_method == 'largest':
        # Select largest cluster (excluding noise)
        largest_size = 0
        selected_label = -1
        for name, cluster_data in clusters.items():
            if name != 'noise' and cluster_data['size'] > largest_size:
                largest_size = cluster_data['size']
                selected_label = int(name.split('_')[1])
        
        selected_stats = clusters[f'cluster_{selected_label}'] if selected_label != -1 else {}
    
    elif selection_method == 'most_compact':
        # Select most compact cluster (smallest spread)
        min_spread = np.inf
        selected_label = -1
        for name, cluster_data in clusters.items():
            if name != 'noise' and cluster_data['spread'] < min_spread:
                min_spread = cluster_data['spread']
                selected_label = int(name.split('_')[1])
        
        selected_stats = clusters[f'cluster_{selected_label}'] if selected_label != -1 else {}
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return selected_label, selected_stats


def plot_translation_clusters(
        tr_preds: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        tr_true: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, Any]] = None,
        method: str = 'tsne',
        tsne_params: Dict[str, Any] = None,
        pca_params: Dict[str, Any] = None,
        title: str = "Translation Clusters",
        figsize: Tuple[int, int] = (12, 8),
        show_centers: bool = True,
        show_best_point: bool = True,
        ) -> plt.Figure:
    """
    Create comprehensive visualization of translation clusters.
    
    Parameters:
    -----------
    tr_preds : np.ndarray, shape (n_samples, 3)
        Translation predictions
    labels : np.ndarray, shape (n_samples,)
        Cluster labels
    scores : np.ndarray, shape (n_samples,)
        Confidence scores
    tr_true : np.ndarray, shape (3,), optional
        True translation
    stats : dict, optional
        Cluster statistics
    method : str
        Visualization method: 'tsne', 'pca'
    tsne_params : dict, optional
        t-SNE parameters
    pca_params : dict, optional
        PCA parameters
    title : str
        Plot title
    figsize : tuple
        Figure size
    show_centers : bool
        Whether to show cluster centers
    show_best_point : bool
        Whether to highlight best point
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if tsne_params is None:
        tsne_params = {
            'n_components': 2,
            'perplexity': min(30, len(tr_preds) // 4),
            'early_exaggeration': 12.0,
            'learning_rate': 'auto',
            'max_iter': 1000,
            'n_iter_without_progress': 300,
            'min_grad_norm': 1e-07,
            'metric': 'euclidean',
            'init': 'pca',
            'random_state': 0xC0FFEE,
            'method': 'barnes_hut',
            'angle': 0.5,
            'n_jobs': -1
        }
    
    if pca_params is None:
        pca_params = {
            'n_components': 3,
            'random_state': 0xC0FFEE
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for visualization
    if method == 'tsne':
        # Combine predictions and true point for t-SNE
        combined_data = tr_preds.copy()
        if tr_true is not None:
            combined_data = np.vstack([combined_data, tr_true.reshape(1, -1)])
        
        # Fit t-SNE
        tsne = TSNE(**tsne_params)
        tsne_results = tsne.fit_transform(combined_data)
        
        # Separate results
        preds_2d = tsne_results[:-1] if tr_true is not None else tsne_results
        true_2d = tsne_results[-1] if tr_true is not None else None
        
        x_label, y_label = 't-SNE 1', 't-SNE 2'
        
    elif method == 'pca':
        # Fit PCA
        pca = PCA(**pca_params)
        preds_2d = pca.fit_transform(tr_preds)
        
        if tr_true is not None:
            true_2d = pca.transform(tr_true.reshape(1, -1))[0]
        
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
        
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'x': preds_2d[:, 0],
        'y': preds_2d[:, 1],
        'score': scores,
        'cluster': labels
    })
    
    # Plot clusters
    scatter = sns.scatterplot(
        data=plot_data,
        x='x', y='y',
        hue='score',
        style='cluster',
        palette='viridis',
        alpha=0.7,
        ax=ax
    )
    
    # Plot true translation if available
    if tr_true is not None and true_2d is not None:
        ax.scatter(true_2d[0], true_2d[1], c='red', s=200, marker='*', 
                   label='True', zorder=5)
    
    # Show cluster centers if available
    if show_centers and stats is not None:
        for cluster_name, cluster_data in stats['clusters'].items():
            if cluster_name != 'noise':
                center = cluster_data['center_median']
                if method == 'tsne':
                    # Project center to t-SNE space (approximate)
                    center_2d = np.mean(preds_2d[labels == cluster_data['indices'][0]], axis=0)
                else:  # PCA
                    center_2d = center[:2]  # Use first two components
                
                ax.scatter(center_2d[0], center_2d[1], c='orange', s=80, 
                           marker='h', label=f'Center {cluster_name}', zorder=4)
    
    # Highlight best point if available
    if show_best_point and stats is not None and stats['total_best_point'] is not None:
        best_idx = stats['total_best_point']['index']
        best_2d = preds_2d[best_idx]
        ax.scatter(best_2d[0], best_2d[1], c='gold', s=150, marker='d', 
                   label='Best Point', zorder=6, edgecolor='black', linewidth=2)
    
    # Add confidence ellipse if enough points
    if len(tr_preds) > 10:
        cov = np.cov(preds_2d.T)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        
        ellipse = Ellipse(
            xy=(np.mean(preds_2d[:, 0]), np.mean(preds_2d[:, 1])),
            width=lambda_[0]*2, height=lambda_[1]*2,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            edgecolor='r', fc='None', lw=2, linestyle='--'
        )
        ax.add_patch(ellipse)
        
        # Add spread information
        ax.text(0.05, 0.95, f'Spread: {np.mean(lambda_):.2f}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Move legend outside
    sns.move_legend(scatter, 'center left', bbox_to_anchor=(1, 0.5), 
                    ncol=1, frameon=False)
    
    plt.tight_layout()
    return fig


def comprehensive_translation_analysis(
        uid_data: Dict[str, Any],
        uid: str,
        clustering_method: str = 'dbscan',
        selection_method: str = 'best_score',
        plot_results: bool = True,
        print_summary: bool = True,
        fig_savepath: Optional[str] = None,
        **clustering_kwargs
        ) -> Dict[str, Any]:
    """
    Comprehensive translation clustering and analysis.
    
    Parameters:
    -----------
    uid_data : dict
        Data for a single complex
    supplement_data : dict
        Supplementary data
    uid : str
        Complex identifier
    clustering_method : str
        Clustering algorithm to use
    selection_method : str
        Method for selecting best cluster
    plot_results : bool
        Whether to create plots
    print_summary : bool
        Whether to print summary statistics
    **clustering_kwargs
        Additional parameters for clustering
        
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    # Extract data
    samples = uid_data['sample_metrics']
    if len(samples) == 0:
        return {'error': 'No samples available'}
    
    tr_preds = np.array([s['tr_pred'] for s in samples])
    scores = np.array([s['error_estimate_0'] for s in samples])
    
    # Get true translation if available
    tr_true = None
    if 'true_pos' in uid_data:
        tr_true = uid_data['true_pos'].mean(0)
    
    # Perform clustering
    labels, cluster_info = cluster_translations(
        tr_preds=tr_preds, 
        scores=scores, 
        method=clustering_method, 
        **clustering_kwargs
        )
    
    # Calculate statistics
    stats = calculate_cluster_statistics(
        tr_preds=tr_preds, 
        scores=scores, 
        labels=labels, 
        tr_true=tr_true
        )
    
    # Select best cluster
    selected_label, selected_stats = select_best_cluster(stats=stats, selection_method=selection_method)
    
    # Create plots if requested
    fig = None
    if plot_results:
        fig = plot_translation_clusters(
            tr_preds=tr_preds, 
            labels=labels, 
            scores=scores, 
            tr_true=tr_true, 
            stats=stats,
            method='tsne',
            title=f"Translation Clusters - {uid}"
            )
        if fig_savepath is not None:
            fig.savefig(os.path.join(fig_savepath, f"{uid}_tr_clusters.png"), dpi=300)
    
    # Print summary if requested
    if print_summary:
        _print_cluster_summary(
            uid=uid, 
            stats=stats, 
            selected_label=selected_label, 
            selected_stats=selected_stats
            )
    
    # Compile results
    results = {
        'uid': uid,
        'labels': labels,
        'cluster_info': cluster_info,
        'all_stats': stats,
        'selected_cluster': {
            'label': selected_label,
            'stats': selected_stats
            },
        'figure': fig,
        'tr_preds': tr_preds,
        'scores': scores,
        'tr_true': tr_true
        }
    
    return results


def _print_cluster_summary(
        uid: str,
        stats: Dict[str, Any],
        selected_label: int,
        selected_stats: Dict[str, Any]
        ) -> None:
    """Print summary of clustering results."""
    print(f"\n=== Translation Clustering Summary for {uid} ===")
    
    overall = stats['overall_stats']
    print(f"Samples: {overall['n_samples']}")
    print(f"Clusters: {overall['n_clusters']}")
    print(f"Noise points: {overall['n_noise']}")
    print(f"Score range: {overall['score_mean']:.3f} ± {overall['score_std']:.3f}")
    
    if selected_label != -1 and selected_stats:
        print(f"\nSelected cluster {selected_label}:")
        print(f"\tSize: {selected_stats['size']}")
        print(f"\tScore: {selected_stats['score_median']:.3f} (median)")
        print(f"\tSpread: {selected_stats['spread']:.3f}")
        
        if 'center_error_median' in selected_stats:
            print(f"\tCenter error: {selected_stats['center_error_median']:.3f}")
        
        if selected_stats['best_point'] is not None:
            best = selected_stats['best_point']
            print(f"\tBest point score: {best['score']:.3f}")
            if 'error' in best:
                print(f"\tBest point error: {best['error']:.3f}")
    
    print("=" * 50)


def batch_translation_analysis(
        preds: Dict[str, Any],
        clustering_method: str = 'dbscan',
        selection_method: str = 'best_score',
        plot_first_n: int = 5,
        **clustering_kwargs
        ) -> Dict[str, Any]:
    """
    Perform translation clustering analysis on all complexes.
    
    Parameters:
    -----------
    preds : dict
        All prediction data
    supplement_data : dict
        Supplementary data
    clustering_method : str
        Clustering algorithm
    selection_method : str
        Best cluster selection method
    plot_first_n : int
        Number of first complexes to plot
    **clustering_kwargs
        Additional clustering parameters
        
    Returns:
    --------
    all_results : dict
        Results for all complexes
    """
    all_results = {}
    
    for i, (uid, uid_data) in enumerate(tqdm(preds.items(), desc="Analyzing translations")):
        plot_results = i < plot_first_n
        print_summary = i < 10  # Print summary for first 10
        
        results = comprehensive_translation_analysis(
            uid_data=uid_data, 
            uid=uid,
            clustering_method=clustering_method,
            selection_method=selection_method,
            plot_results=plot_results,
            print_summary=print_summary,
            **clustering_kwargs
            )
        
        all_results[uid] = results
    
    return all_results
