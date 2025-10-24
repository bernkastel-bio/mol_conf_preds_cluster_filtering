import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch # type: ignore

import signal
from contextlib import contextmanager

from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering

from scipy.stats import shapiro
from scipy.stats import median_abs_deviation
from scipy.stats import vonmises
from scipy.stats import circmean, circstd

from astropy.stats import rayleightest # type: ignore
from astropy.stats import kuiper # type: ignore

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer

from icecream import ic
ic.configureOutput(includeContext=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

from typing import List, Tuple, Union, Optional, Any, Dict


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_symmetry_rmsd_with_isomorphisms(
        coords1: np.ndarray, 
        coords2: np.ndarray, 
        isomorphisms: List[Tuple],
        time_limit_sec: float = 1
        ):
    with time_limit(time_limit_sec):
        
        assert coords1.shape == coords2.shape

        n = coords1.shape[0]

        # Minimum result
        # Squared displacement (not minimize) or RMSD (minimize)
        min_result = np.inf

        # Loop over all graph isomorphisms to find the lowest RMSD
        for idx1, idx2 in isomorphisms:
            # Use the isomorphism to shuffle coordinates around (from original order)
            c1i = coords1[idx1, :]
            c2i = coords2[idx2, :]

            # Compute square displacement
            # Avoid dividing by n and an expensive sqrt() operation
            result = np.sum((c1i - c2i) ** 2)

            if result < min_result:
                min_result = result

        # Compute actual RMSD from square displacement
        min_result = np.sqrt(min_result / n)

        # Return the actual RMSD
        return min_result


def compute_signed_dihedral_angles_vectorized(
        positions: np.ndarray
        ):
    """
    Compute signed dihedral angles for multiple sets of atoms in a vectorized way.
    
    Parameters:
    -----------
    positions: np.ndarray or torch.Tensor, shape (B, 4, 3)
        Positions of B sets of 4 atoms each: (n0, start, end, n1) in 3D space.
        
    Returns:
    --------
    angles: np.ndarray or torch.Tensor, shape (B,)
        The signed dihedral angles in radians.
    """
    # Extract the positions of each atom in each set
    p0 = positions[:, 0]  # n0 positions
    p1 = positions[:, 1]  # start positions
    p2 = positions[:, 2]  # end positions
    p3 = positions[:, 3]  # n1 positions
    
    # Calculate bond vectors
    b1 = p1 - p0  # n0 -> start
    b2 = p2 - p1  # start -> end
    b3 = p3 - p2  # end -> n1
    
    # Normalize b2
    if isinstance(positions, torch.Tensor):
        b2_normalized = b2 / torch.norm(b2, dim=1, keepdim=True)
        
        # Calculate normal vectors to the planes
        n1 = torch.cross(b1, b2, dim=1)
        n1_normalized = n1 / torch.norm(n1, dim=1, keepdim=True)
        
        n2 = torch.cross(b2, b3, dim=1)
        n2_normalized = n2 / torch.norm(n2, dim=1, keepdim=True)
        
        # Calculate the orthogonal vector to n1 in the plane defined by b2
        m1 = torch.cross(n1_normalized, b2_normalized, dim=1)
        
        # Calculate cosine and sine
        x = torch.sum(n1_normalized * n2_normalized, dim=1)
        y = torch.sum(m1 * n2_normalized, dim=1)
        
        # Calculate dihedral angle
        angles = torch.atan2(y, x)
    else:
        b2_normalized = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
        
        # Calculate normal vectors to the planes
        n1 = np.cross(b1, b2)
        n1_normalized = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
        
        n2 = np.cross(b2, b3)
        n2_normalized = n2 / np.linalg.norm(n2, axis=1, keepdims=True)
        
        # Calculate the orthogonal vector to n1 in the plane defined by b2
        m1 = np.cross(n1_normalized, b2_normalized)
        
        # Calculate cosine and sine
        x = np.sum(n1_normalized * n2_normalized, axis=1)
        y = np.sum(m1 * n2_normalized, axis=1)
        
        # Calculate dihedral angle
        angles = np.arctan2(y, x)
    
    return -angles


def find_rigid_alignment(
        pos_a, 
        pos_b
        ):
    """
    Borrowed and slightly modified from
    https://gist.github.com/bougui505/23eb8a39d7a601399edc7534b28de3d4

    Outputs rot and tr (with fixed tor components)
    """
    a_mean = pos_a.mean(0)
    b_mean = pos_b.mean(0)
    a_centered = pos_a - a_mean
    b_centered = pos_b - b_mean
    # Covariance matrix
    cov_mat = a_centered.T @ b_centered
    if isinstance(pos_a, torch.Tensor):
        U, _, Vt = torch.linalg.svd(cov_mat)
        V = Vt.T
        det = torch.linalg.det(V @ U.T)
    else:
        U, _, Vt = np.linalg.svd(cov_mat)
        V = Vt.T
        det = np.linalg.det(V @ U.T)

    # Ensure proper rotation by checking determinant
    if det < 0:
        V[:, -1] = -V[:, -1]  # Flip the last column of V
    # Rotation matrix (now guaranteed to be proper rotation)
    rot = V @ U.T
    # Translation vector
    tr = b_mean
    return rot, tr


def get_torsion_angles(pos, bond_atoms_for_angles):
    # Create a batch of all atom quartets for dihedral calculations
    n0 = bond_atoms_for_angles['neighbor_of_start']
    start = bond_atoms_for_angles['start']
    end = bond_atoms_for_angles['end']
    n1 = bond_atoms_for_angles['neighbor_of_end']
    
    # Stack the positions to form a batch
    if isinstance(pos, torch.Tensor):
        atom_quartets = torch.stack([
            pos[n0], pos[start], pos[end], pos[n1]
        ], dim=1)
    else:
        atom_quartets = np.stack([
            pos[n0], pos[start], pos[end], pos[n1]
        ], axis=1)
    
    # Calculate all dihedral angles at once
    angles = compute_signed_dihedral_angles_vectorized(atom_quartets)
    
    # Fix the angles based on bond periods
    angles = fix_torsion_angles(angles, bond_atoms_for_angles['bond_periods'])
    return angles


def fix_torsion_angles(
        angles: np.ndarray, 
        bond_periods: np.ndarray
        ) -> np.ndarray:
    return (angles + bond_periods / 2) % bond_periods - bond_periods / 2


def compute_angle_MAE_batch(
        pred_angles: np.ndarray, 
        true_angles: np.ndarray, 
        bond_periods: np.ndarray
        ) -> float:
    # Compute the difference between the predicted and true torsion angles
    if len(bond_periods) > 0:
        diff = (pred_angles - true_angles + \
                bond_periods / 2) % bond_periods - bond_periods / 2
        # need to average over the bond dimension
        diff = np.abs(diff)
        angle_err = np.mean(diff, axis=-1)
    else:
        angle_err = 0
    return angle_err


def test_vonmises_fit(
        angles_rad: np.ndarray, 
        bond_period: float
        ) ->Tuple[Any]:
    # Fit von Mises
    kappa, loc, scale = vonmises.fit(angles_rad, fscale=1)
    # Generate CDFs
    sorted_angles = np.sort(angles_rad)
    empirical_cdf = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
    theoretical_cdf = vonmises.cdf(sorted_angles, kappa, loc=loc, scale=scale)
    theoretical_pdf = vonmises.pdf(sorted_angles, kappa, loc=loc, scale=scale)
    best_angle = sorted_angles[np.argmax(theoretical_pdf)]
    
    shifted_angles = (sorted_angles - loc - bond_period / 2) % bond_period  - bond_period / 2
    median = np.median(shifted_angles) + loc #- bond_period / 2

    # Kuiper's test
    D, p_value = kuiper(sorted_angles, lambda x: vonmises.cdf(x, kappa, loc=loc, scale=scale))    
    return p_value, (best_angle, loc, median), theoretical_pdf, (kappa, loc, scale)


def analyze_circular_distribution(
        angles_rad, 
        true_rad, 
        bond_period, 
        eps=0.3, 
        min_samples=5, 
        normality_p=0.05
        ):
    # Map to unit circle
    X = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
    # Cluster
    clustering = AffinityPropagation().fit(X)
    # clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    
    # Find largest cluster (excluding noise)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        # "1 no clear normal distribution"
        return None
    largest_cluster_label = unique[np.argmax(counts)]
    largest_cluster = angles_rad[labels == largest_cluster_label]
        
    # Check if largest cluster is dominant
    if len(largest_cluster) < 0.5 * len(angles_rad):
        # "2 no clear normal distribution"
        return None
    
    # Test for circular normality
    p, (best_angle, loc, median), theoretical_pdf, params = test_vonmises_fit(largest_cluster, bond_period)    
    test_passed = ((p < normality_p) and (params[0] > 10)) or (params[0] > 100)
    
    if test_passed:
        angles_diff = (best_angle - true_rad + bond_period / 2) % bond_period - bond_period / 2
        angles_diff = np.abs(angles_diff)
    
    if test_passed:  # Not significantly different from von Mises
        return best_angle
    
    return None

# TODO: add test_vonmises_fit for best cluster if clusters found
# if test passed we consider median as best angle
# if not, replace the other, non-best-cluster-values with best cluster median
def select_and_update_torsion_angles(
        torsion_angles_pred: np.ndarray,
        bond_analyses: Dict[int, Any],
        bond_periods: np.ndarray,
        scores: Optional[np.ndarray] = None,
        normality_p: float = 0.05
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Given predicted torsion angles and per-bond analysis (as returned by
    `tor_clustering_bonds.analyze_all_bonds_separately`), produce a copied
    updated torsion-angle array using the following rules:

    1) If a bond has real clusters (labels != -1, n_uniq_labels > 1) and 
       score-error correlation is >0.5, pick the selected/best cluster 
       (based on `bond_analyses[b]['statistics']['best_cluster_label']`
       or falling back to the largest cluster). Choose the best exemplar 
       in that cluster using (in order of preference):
         - cluster_stats[best]['min_score_angle']
         - cluster_stats[best]['min_error_angle']
         - cluster_stats[best]['mean_angle']
       Replace all sample angles for that bond with the chosen angle.

    2) If there are no clusters (only noise), test the bond angles for a
       von mises fit (using `test_vonmises_fit`). If the von mises fit
       indicates a concentrated distribution (kappa large enough and the
       Kuiper test threshold), consider the distribution normal.
       In that case detect outliers in the rotated (centered) linear frame
       using the boxplot rule [Q1 - IQR, Q3 + IQR] and replace outliers by the
       circular median value. If not von mises-like, leave angles unchanged.
    
    The function returns the updated angles array and an info dict with
    per-bond chosen angles and outlier indices.

    Returns:
    --------
    updated_angles : np.ndarray, shape (n_samples, n_bonds)
    info : dict with keys:
        'chosen_angles' : list of per-bond chosen angles or None
        'outlier_indices' : dict bond_idx -> list of indices replaced
    """
    updated = np.array(torsion_angles_pred, copy=True)
    n_samples, n_bonds = updated.shape

    chosen_angles = [None] * n_bonds
    outlier_indices = {}

    for bond_idx in range(n_bonds):
        angles = updated[:, bond_idx]
        period = float(bond_periods[bond_idx]) if bond_periods is not None else 2 * np.pi

        # Normalize to canonical range for working ([-period/2, period/2])
        angles_norm = (angles + period / 2) % period - period / 2

        # Retrieve analysis for bond if provided
        analysis = bond_analyses.get(bond_idx, None) if bond_analyses is not None else None

        labels = None
        stats = None
        if analysis is not None:
            labels = analysis.get('cluster_labels', None)
            stats = analysis.get('statistics', None)

        # Determine if there are bona-fide clusters (exclude noise label -1)
        has_clusters = False
        if labels is not None:
            uniq = np.unique(labels)
            non_noise = uniq[uniq != -1] if len(uniq) > 0 else np.array([])
            has_clusters = len(non_noise) > 0

        if has_clusters and stats['n_clusters'] != 1:
            # Prefer the cluster selected by cluster analysis
            best_lbl = stats.get('best_cluster_label', None)
            cluster_stats = stats.get('cluster_stats', {})
            
            # if stats['score_error_correlation'] >= 0.5:
            #     if best_lbl is None or best_lbl not in cluster_stats:
            #         # fallback to largest cluster
            #         if cluster_stats:
            #             best_lbl = max(cluster_stats.items(), key=lambda kv: kv[1].get('size', 0))[0]
            #         else:
            #             best_lbl = None
            # else:
            #     if cluster_stats:
            #         best_lbl = max(cluster_stats.items(), key=lambda kv: kv[1].get('size', 0))[0]
            #     else:
            #         best_lbl = None
            
            # Test selected cluster distribution if it is normal
            best_cluster_angles = angles[labels == best_lbl]
            try:
                pval, (best_angle_vm, loc, median_vm), pdf, params = test_vonmises_fit(best_cluster_angles, period)
                kappa = params[0]
                vm_ok = ((pval < normality_p) and (kappa > 10)) or (kappa > 100)
            except Exception:
                vm_ok = False
                best_angle_vm = None
                median_vm = None
            
            if vm_ok:

                chosen_angle = None
                if best_lbl is not None:
                    cs = cluster_stats.get(best_lbl, {})
                    # prefer min_score_angle, then min_error_angle, then mean_angle
                    chosen_angle = cs.get('mean_angle')
                    # chosen_angle = cs.get('min_score_angle')
                    # if chosen_angle is None:
                    #     chosen_angle = cs.get('min_error_angle')
                    # if chosen_angle is None:
                    #     chosen_angle = cs.get('mean_angle')

                # Final fallback: compute circular mean of cluster members
                # if chosen_angle is None and labels is not None and best_lbl is not None:
                #     mask = labels == best_lbl
                #     if mask.sum() > 0:
                #         member_angles = angles[mask]
                #         c_cos = np.mean(np.cos(member_angles))
                #         c_sin = np.mean(np.sin(member_angles))
                #         chosen_angle = float(np.arctan2(c_sin, c_cos))

                if chosen_angle is not None:
                    # Normalize chosen angle and set for all samples
                    chosen_angle = (float(chosen_angle) + period / 2) % period - period / 2
                    updated[:, bond_idx] = chosen_angle
                    chosen_angles[bond_idx] = chosen_angle
                else:
                    chosen_angles[bond_idx] = None
            else:
                center = median_vm if median_vm is not None else (best_angle_vm if best_angle_vm is not None else 0.0)
                center = (float(center) + period / 2) % period - period / 2
                mask_replacement = labels != best_lbl
                updated[mask_replacement, bond_idx] = center

        else:
            # No clusters (only noise) or single cluster — try vonmises fit to check for concentrated data
            try:
                pval, (best_angle_vm, loc, median_vm), pdf, params = test_vonmises_fit(angles, period)
                kappa = params[0]
                vm_ok = ((pval < normality_p) and (kappa > 10)) or (kappa > 100)
            except Exception:
                vm_ok = False
                best_angle_vm = None
                median_vm = None

            if vm_ok:
                # center distribution around its circular median (use median_vm if available)
                center = median_vm if median_vm is not None else (best_angle_vm if best_angle_vm is not None else 0.0)
                shifted = (angles - center + period / 2) % period - period / 2

                q1 = np.percentile(shifted, 25)
                q3 = np.percentile(shifted, 75)
                iqr = q3 - q1
                lower = q1 - iqr
                upper = q3 + iqr

                mask_outliers = (shifted < lower) | (shifted > upper)
                if np.any(mask_outliers):
                    # Replace outliers with circular median (computed from non-outliers or full set)
                    if np.any(~mask_outliers):
                        median_shift = np.median(shifted[~mask_outliers])
                    else:
                        median_shift = np.median(shifted)

                    replacement = (median_shift + center + period / 2) % period - period / 2
                    updated[mask_outliers, bond_idx] = replacement
                    chosen_angles[bond_idx] = replacement
                    outlier_indices[bond_idx] = np.where(mask_outliers)[0].tolist()
                else:
                    chosen_angles[bond_idx] = None
            else:
                # Cannot confidently change noisy non-normal distribution
                chosen_angles[bond_idx] = None

    info = {
        'chosen_angles': chosen_angles,
        'outlier_indices': outlier_indices
    }

    # Ensure result is normalized (again)
    for bond_idx in range(n_bonds):
        period = float(bond_periods[bond_idx]) if bond_periods is not None else 2 * np.pi
        updated[:, bond_idx] = (updated[:, bond_idx] + period / 2) % period - period / 2

    return updated, info


def apply_tor_changes_to_pos(
        pos, 
        rotatable_bonds, 
        mask_rotate, 
        torsion_updates, 
        is_reverse_order, 
        bond_properties_for_angles=None, 
        shift_center_back=True
        ) -> np.ndarray:
    """
    Apply torsion updates to the positions of atoms in a sample in-place.

    Parameters:
    ----------
    pos : Union[np.ndarray, torch.Tensor]
        The positions of atoms in the sample, shape (num_atoms, 3).
    rotatable_bonds : Union[np.ndarray, torch.Tensor]
        Rotatable bonds in the sample, shape (num_rotatable_bonds, 2). Each bond is represented by
        two indices: (atom1, atom2).
    mask_rotate : Union[np.ndarray, torch.Tensor]
        Mask indicating which atoms to rotate for each bond, shape (num_rotatable_bonds, num_atoms).
    torsion_updates : Union[np.ndarray, torch.Tensor]
        Torsion updates to apply to each rotatable bond, shape (num_rotatable_bonds,).

    Returns:
    -------
    Union[np.ndarray, torch.Tensor]
        The updated positions of atoms in the sample.
    """
    is_torch = isinstance(pos, torch.Tensor)

    if len(rotatable_bonds) == 0:
        return pos, None

    if rotatable_bonds.shape[1] != 2:
        raise ValueError('A wrong format of rotational bonds array!')

    num_rotatable_bonds = rotatable_bonds.shape[0]

    if is_reverse_order:
        range_for_rot_bonds = range(num_rotatable_bonds - 1, -1, -1)
    else:
        range_for_rot_bonds = range(num_rotatable_bonds)

    # compute initial ligand center
    pos_mean = pos.mean(0)[None, :]

    for idx_rot_bond in range_for_rot_bonds:
        u = rotatable_bonds[idx_rot_bond, 0]
        v = rotatable_bonds[idx_rot_bond, 1]
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards

        if is_torch:
            # Rotate v:
            rot_vec = rot_vec * torsion_updates[idx_rot_bond] / torch.linalg.norm(rot_vec)
            rot_mat = rotvec_to_rotmat(rot_vec) # type: ignore
            mask = mask_rotate[idx_rot_bond].bool()
            pos[mask] = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]
        else:
            # Rotate v:
            rot_vec = rot_vec * torsion_updates[idx_rot_bond] / np.linalg.norm(rot_vec)
            rot_mat = rotvec_to_rotmat(torch.tensor(rot_vec, dtype=torch.float)).numpy() # type: ignore
            mask = mask_rotate[idx_rot_bond].astype(bool)
            pos[mask] = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]

    # shift to the initial center
    if shift_center_back:
        pos = pos - pos.mean(0)[None, :] + pos_mean

    return pos


def rotvec_to_rotmat(
        rot_vec: np.ndarray
        ) -> np.ndarray:  
    """
    Convert a rotation vector (axis * angle) to a rotation matrix using Rodrigues' formula.
    Supports numpy arrays and torch tensors.
    rot_vec: shape (3,) or (N,3)
    Returns a 3x3 rotation matrix (numpy) or Nx3x3 (torch) depending on input type.
    """
    # If torch tensor
    if isinstance(rot_vec, torch.Tensor):
        vec = rot_vec
        theta = torch.linalg.norm(vec)
        if theta == 0:
            return torch.eye(3)
        k = vec / theta
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=vec.dtype)
        R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R

    # numpy path
    vec = np.array(rot_vec, dtype=float)
    theta = np.linalg.norm(vec)
    if theta == 0:
        return np.eye(3)
    k = vec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def plot_errors(
    rmsds, 
    tr_errors, 
    rot_sims, 
    tor_errs, 
    score_estimates, 
    is_pb_valid, 
    name
    ):
    # Create figure with 2 horizontal subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Create size array: larger dots for PB valid samples
    sizes = np.where(is_pb_valid, 50, 20)  # 50 for PB valid, 20 for invalid

    ax1.scatter(score_estimates, rmsds, 
                c=is_pb_valid,
                cmap='winter',  # Color map
                alpha=0.5)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('RMSD')
    ax1.set_title(f'{name}, RMSD')

    ax2.scatter(score_estimates, tr_errors,
                c=is_pb_valid,
                cmap='winter',  # Color map
                alpha=0.5)
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Tr err')
    ax2.set_title(f'{name}: Tr err')
    
    ax3.scatter(score_estimates, rot_sims,
                c=is_pb_valid,
                cmap='winter',  # Color map
                alpha=0.5)
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Rot sim')
    ax3.set_title(f'{name}: Rot sim')
    
#     ax4.scatter(score_estimates, tor_errs,
#                 c=is_pb_valid,
#                 cmap='winter',  # Color map
#                 alpha=0.5)
#     ax4.set_xlabel('Score')
#     ax4.set_ylabel('Tor err')
#     ax4.set_title(f'{name}: Tor err')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    
def err_print_fn(
        name, 
        current_errs, 
        current_errs_pb, 
        thresholds
        ):
    # Build the format string dynamically based on thresholds
    line = f"{name:<30}\t"
    
    # Add percentage below thresholds for regular errors
    for threshold in thresholds:
        line += f"{(current_errs < threshold).mean():.3f}\t"
    
    # Add percentage below thresholds for PoseBusters valid errors
    for threshold in thresholds:
        line += f"{((current_errs < threshold) & current_errs_pb).mean():.3f}\t"
    
    # Add mean and median
    line += f"{np.mean(current_errs):.3f}\t"
    line += f"{np.median(current_errs):.3f}"
    
    print(line)
    

def pretty_print_of_metrics(all_errs):
#     for metric_type in all_errs[list(all_errs.keys())[0]].keys():
    thresholds_for_metrics = {
        'RMSD': [2, 5],
        'Tr': [1, 2, 5],
        'Rot': [],
        'Tor': [],
      }
    for metric_type, thresholds in thresholds_for_metrics.items():
        print('Metric:', metric_type)
        
        # Print column headers with same indentation
        header = f"{'Aggregation':<30}\t"
        for threshold in thresholds:
            header += f"%<{threshold}Å\t"
        for threshold in thresholds:
            header += f"%<{threshold}Å&PB\t"
        header += "mean\tmedian"
        print(header)
        
        for agg_type in all_errs[list(all_errs.keys())[0]][metric_type].keys():
            current_errs = np.array([item[metric_type][agg_type][0] for item in all_errs.values()])
            current_errs_pb = np.array([item[metric_type][agg_type][1] for item in all_errs.values()])
            err_print_fn(agg_type, current_errs, current_errs_pb, thresholds)
        print()


def plot_torsion_data(torsion_angles_true, torsion_angles_pred):
    for i in range(len(torsion_angles_true)):
        true_rad = torsion_angles_true[i]
        pred_rads = torsion_angles_pred[:, i]
        plt.axvline(x=true_rad, color='green', label='true')
        plt.hist(pred_rads, bins=50, density=False, color='orange')
        plt.title(f'Angle {i}')
        plt.show()