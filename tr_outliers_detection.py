import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.impute import SimpleImputer
from scipy.stats import median_abs_deviation

from typing import Dict, Any


def detect_outliers_consensus(metrics_dict, contamination=0.1):
    """
    Consensus outlier detection using multiple metrics and algorithms.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing arrays of metrics for all predictions
    contamination : float
        Expected proportion of outliers (0.05-0.2)
    
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean mask where True indicates outliers
    """
    # Extract metrics
    scores = metrics_dict['scores']
    pb_valid = metrics_dict['pb_valid']
    
    n_samples = len(scores)
    
    # Create feature matrix (normalize all metrics to be "higher is worse")
    X = np.column_stack([
        # rmsds,  # Higher is worse
        # tr_errs,  # Higher is worse  
        # 1 - rot_sims,  # Convert similarity to error (higher is worse)
        # tor_errs,  # Higher is worse
        # scores - np.min(scores),  # Normalize scores (higher is worse)
        scores
    ])
    
    # TODO uspervised outlier detection without metrics from true
    # mask = np.any(np.isnan(X), axis=1)
    # X = X[~mask]
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = imp.fit(X)
    # X = imp.transform(X)
    
    # Standardize features
    # X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = imp.fit(X_standardized)
    # X_standardized = imp.transform(X_standardized)
    
    # Use multiple outlier detection methods
    methods = {
        'isolation_forest': IsolationForest(contamination=contamination, random_state=42),
        'local_outlier_factor': LocalOutlierFactor(n_neighbors=min(20, n_samples//5), contamination=contamination),
        }
    
    outlier_votes = np.zeros(n_samples)
    
    try:
        for name, detector in methods.items():
            if name == 'local_outlier_factor':
                labels = detector.fit_predict(X)
            else:
                labels = detector.fit_predict(X)

            # Convert to binary (1 for outlier, 0 for inlier)
            outlier_votes += (labels == -1).astype(int)

        # Consensus: consider outlier if majority of methods agree
        outlier_mask = outlier_votes == 2

    except Exception as e:
        print(X)
        print(e)
    
    return outlier_mask


def detect_cluster_outliers(
        tr_preds: np.ndarray, 
        scores: np.ndarray, 
        DBSCAN_params: Dict[str, Any] = {
            'eps': 2.0, 
            'min_samples': 5
            },
        HDBSCAN_params: Dict[str, Any] = {
            'min_cluster_size': 3,
            'min_samples': None,
            'cluster_selection_epsilon': 0.0,
            'alpha': 1.0,
            'algorithm': 'auto',
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
            'store_centers': 'centroid',
            'n_jobs': -1
            },
        AffinityProp_params: Dict[str, Any] = {
            'damping': 0.5, 
            'max_iter': 200, 
            'convergence_iter': 15, 
            'copy': True, 
            'preference': None, 
            'affinity': 'euclidean'
            },
        min_points: int = 2
        ):
    """
    Detect outliers based on clustering in translation space.
    
    Parameters:
    -----------
    tr_preds : np.ndarray, shape (n_samples, 3)
        Predicted translation vectors
    scores : np.ndarray, shape (n_samples,)
        Confidence scores
    
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean mask where True indicates outliers
    """
    
    # Cluster in translation space
    clustering = HDBSCAN(**HDBSCAN_params).fit(tr_preds)
    # clustering = DBSCAN(**DBSCAN_params).fit(tr_preds)
    # clustering = AffinityPropagation(**AffinityProp_params).fit(tr_preds)
    labels = clustering.labels_
    
    outlier_mask = np.zeros(len(tr_preds), dtype=bool)
    
    # Analyze each cluster
    unique_labels = set(labels)
    
    for n, label in enumerate(unique_labels):
        if label == -1:  # Noise points
            best_cluster_label = -1
            cluster_mask = labels == label
            # Consider all noise points as potential outliers
            outlier_mask[cluster_mask] = True
        else:
            cluster_mask = labels == label
            # cluster_size = np.sum(cluster_mask)
            
            # Check if cluster has poor score statistics
            cluster_scores = scores[cluster_mask]
            total_scores_iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
            third_quartile_score = np.percentile(scores, 75)
            outlier_above_score = third_quartile_score + 1.5 * total_scores_iqr
            # Consider points above the outlier score as outliers
            for i, score in zip(np.where(cluster_mask)[0], cluster_scores):
                if score > outlier_above_score:
                    outlier_mask[i] = True
            
            if n == 0:
                best_cluster_label = label
                best_cluster_score = np.median(cluster_scores)
            else:
                # Compare cluster median score to best cluster
                cluster_median_score = np.median(cluster_scores)
                if cluster_median_score < best_cluster_score:
                    best_cluster_label = label
                    best_cluster_score = cluster_median_score
    
    # TODO add score ranking to select best
    
    return outlier_mask, best_cluster_label


# def detect_statistical_outliers(metrics_dict, z_threshold=2.5):
#     """
#     Detect outliers using Z-scores on individual metrics.
    
#     Parameters:
#     -----------
#     metrics_dict : dict
#         Dictionary containing metric arrays
#     z_threshold : float
#         Z-score threshold for outlier detection
    
#     Returns:
#     --------
#     outlier_mask : np.ndarray
#         Boolean mask where True indicates outliers
#     """
    
#     metrics = ['rmsds', 'tr_errs', 'tor_errs']
#     outlier_masks = []
    
#     for metric_name in metrics:
#         values = metrics_dict[metric_name]
        
#         # Use median and MAD for robustness
#         median = np.median(values)
#         mad = median_abs_deviation(values)
#         if mad == 0:
#             mad = 1e-9
        
#         # Modified Z-score
#         modified_z = 0.6745 * (values - median) / mad  # 0.6745 converts MAD to SD for normal dist
        
#         # Mark as outlier if significantly worse than median
#         outlier_masks.append(modified_z > z_threshold)
    
#     # Convert rotation similarity to error for consistency
#     rot_errors = 1 - metrics_dict['rot_sims']
#     median_rot = np.median(rot_errors)
#     mad_rot = median_abs_deviation(rot_errors)
#     modified_z_rot = 0.6745 * (rot_errors - median_rot) / mad_rot
#     outlier_masks.append(modified_z_rot > z_threshold)
    
#     # Score outliers (higher scores are worse)
#     scores = metrics_dict['scores']
#     median_score = np.median(scores)
#     mad_score = median_abs_deviation(scores)
#     modified_z_score = 0.6745 * (scores - median_score) / mad_score
#     outlier_masks.append(modified_z_score > z_threshold)
    
#     # Combine: outlier if abnormal in multiple metrics
#     outlier_matrix = np.column_stack(outlier_masks)
#     outlier_count = np.sum(outlier_matrix, axis=1)
#     outlier_mask = outlier_count >= 3  # Outlier if abnormal in at least 3 metrics
    
#     return outlier_mask


def comprehensive_outlier_detection(preds, supplement_data, contamination=0.15):
    """
    Comprehensive outlier detection pipeline for molecular docking predictions.
    
    Parameters:
    -----------
    preds : dict
        Predictions dictionary
    supplement_data : dict
        Supplementary data dictionary
    contamination : float
        Expected outlier proportion
    
    Returns:
    --------
    cleaned_preds : dict
        Predictions with outliers removed
    outlier_info : dict
        Information about detected outliers
    """
    
    cleaned_preds = {}
    outlier_info = {}
    best_cluster_labels = {}
    
    for uid, uid_data in preds.items():
        samples = uid_data['sample_metrics']
        
        # Extract metrics
        metrics_dict = {
            'scores': np.array([s['error_estimate_0'] for s in samples]),
            'pb_valid': np.array([s['all_posebusters_filters_passed_count'] == 27 for s in samples])
        }
        
        tr_preds = np.array([s['tr_pred'] for s in samples])
        
        # Apply multiple detection methods
        consensus_mask = detect_outliers_consensus(metrics_dict, contamination)
        cluster_mask, best_cluster_label = detect_cluster_outliers(tr_preds, metrics_dict['scores'], metrics_dict['pb_valid'])

        best_cluster_labels[uid] = best_cluster_label
        
        # Final decision: outlier if detected by at least 2 methods
        final_outlier_mask = (consensus_mask.astype(int) + 
                            cluster_mask.astype(int)) == 2
        
        # Safety check: never remove best-scoring predictions with good physical validity
        best_score_idx = np.argmin(metrics_dict['scores'])
        if metrics_dict['pb_valid'][best_score_idx]:
            final_outlier_mask[best_score_idx] = False
        
        # Keep inliers
        inlier_indices = np.where(~final_outlier_mask)[0]
        cleaned_samples = [samples[i] for i in inlier_indices]
        
        # Store results
        cleaned_preds[uid] = uid_data.copy()
        cleaned_preds[uid]['sample_metrics'] = cleaned_samples
        
        outlier_info[uid] = {
            'n_original': len(samples),
            'n_cleaned': len(cleaned_samples),
            'outlier_indices': np.where(final_outlier_mask)[0],
            'outlier_metrics': {k: v[final_outlier_mask] for k, v in metrics_dict.items()}
        }
    
    return cleaned_preds, outlier_info


def analyze_outlier_effect(original_preds, cleaned_preds, outlier_info):
    """
    Analyze the effect of outlier removal on prediction quality.
    """
    
    print("Outlier Removal Analysis:")
    # print("=" * 50)
    # info_dict = {}
    for uid in original_preds.keys():
        orig_metrics = original_preds[uid]['sample_metrics']
        clean_metrics = cleaned_preds[uid]['sample_metrics']
        info = outlier_info[uid]
        
        # Calculate average metrics
        orig_rmsd = np.mean([s['symm_rmsd'] for s in orig_metrics])
        clean_rmsd = np.mean([s['symm_rmsd'] for s in clean_metrics])
        
        orig_score = np.mean([s['error_estimate_0'] for s in orig_metrics])
        clean_score = np.mean([s['error_estimate_0'] for s in clean_metrics])
        
        orig_pb = np.mean([s['all_posebusters_filters_passed_count'] == 27 for s in orig_metrics])
        clean_pb = np.mean([s['all_posebusters_filters_passed_count'] == 27 for s in clean_metrics])
        
        # info_dict.update({
        #     uid: {
        #         'samples': f'{info['n_original']} → {info['n_cleaned']}', 
        #         'avg rmsd': f'{orig_rmsd:.3f} → {clean_rmsd:.3f}',
        #         'avg score': f'{orig_score:.3f} → {clean_score:.3f}',
        #         'pb valid %': f'{orig_pb*100:.1f}% → {clean_pb*100:.1f}%'
        #         }
        #     })
        
        print(f"{uid}:")
        print(f"  Samples: {info['n_original']} → {info['n_cleaned']} "
              f"({info['n_original'] - info['n_cleaned']} outliers removed)")
        print(f"  Avg RMSD: {orig_rmsd:.3f} → {clean_rmsd:.3f}")
        print(f"  Avg Score: {orig_score:.3f} → {clean_score:.3f}")
        print(f"  PB Valid %: {orig_pb*100:.1f}% → {clean_pb*100:.1f}%")
        print()
        # return info_dict