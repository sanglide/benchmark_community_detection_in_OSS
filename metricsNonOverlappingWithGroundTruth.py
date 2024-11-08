import networkx as nx
import itertools
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.special import comb
from cdlib import evaluation, algorithms


'''
----------------------------------------------------
3.1 Ground-Truth-Based Validation Metrics for Nonoverlapping Community Structure

(1) Purrity: compute_purity(G,partition, ground_truth)
(2) F-measure:compute_f_measure(purity, inverse_purity),  
(3) RI:compute_rand_index(partition, ground_truth),
(4) ARI: compute_ari(G, partition, ground_truth)
(5) NMI: compute_nmi(G, partition, ground_truth)
(6) VI, modified purity, modified ARI and modified NMI:  compute_metrics(G, partition, ground_truth)
(7) Edit Distance: edit_distance_community(G,partition, ground_truth) 

'''

def convert_to_nx_partition(node_clustering):
    partition = {}
    for community_id, community in enumerate(node_clustering.communities):
        partition[community_id]=community
    return partition

def compute_purity(partition, ground_truth):
    """
    Compute the purity and inverse purity for given partition and ground truth.
    """
    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)

    N = sum(len(v) for v in partition.values())
    total_max_overlap = 0
    total_max_overlap_inverse = 0

    # Purity
    for community in partition.values():
        max_overlap = max(len(set(community) & set(gt_community)) for gt_community in ground_truth.values())
        total_max_overlap += max_overlap

    # Inverse Purity
    for gt_community in ground_truth.values():
        max_overlap = max(len(set(gt_community) & set(community)) for community in partition.values())
        total_max_overlap_inverse += max_overlap

    return total_max_overlap / N, total_max_overlap_inverse / N

def compute_rand_index(partition, ground_truth):
    """
    Compute the Rand Index for given partition and ground truth.
    """
    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)
    
    all_nodes = set(itertools.chain.from_iterable(partition.values()))
    tp_plus_fp = sum(len(list(itertools.combinations(community, 2))) for community in partition.values())
    tp_plus_fn = sum(len(list(itertools.combinations(community, 2))) for community in ground_truth.values())

    tp = sum(len(set(community1) & set(community2)) for community1 in partition.values() for community2 in ground_truth.values())

    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = len(list(itertools.combinations(all_nodes, 2))) - tp - fp - fn

    return (tp + tn) / (tp + fp + fn + tn)

def compute_f_measure(purity, inverse_purity):
    """
    Compute the F-measure from purity and inverse purity.
    """
    return (2 * purity * inverse_purity) / (purity + inverse_purity)


def compute_ari(G, partition, ground_truth):
    """
    Compute the Adjusted Rand Index (ARI)
    for given partition and ground truth of a graph G.

    Parameters:
    - G: A networkx graph.
    - partition: A dictionary where keys are community identifiers and values are lists of nodes representing detected communities.
    - ground_truth: A dictionary where keys are community identifiers and values are lists of nodes representing ground-truth communities.

    Returns:
    - ARI: Adjusted Rand Index
    """
    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)
    

    # Create label arrays for ARI and NMI calculations
    node_list = list(G.nodes())
    partition_labels = [None] * len(node_list)
    ground_truth_labels = [None] * len(node_list)

    # Assign labels according to partition
    for label, nodes in partition.items():
        for node in nodes:
            partition_labels[node_list.index(node)] = label

    # Assign labels according to ground truth
    for label, nodes in ground_truth.items():
        for node in nodes:
            ground_truth_labels[node_list.index(node)] = label

    # Compute ARI and NMI
    ari = adjusted_rand_score(ground_truth_labels, partition_labels)

    return ari


def compute_nmi(G, partition, ground_truth):
    """
    Compute Normalized Mutual Information (NMI)
    for given partition and ground truth of a graph G.

    Parameters:
    - G: A networkx graph.
    - partition: A dictionary where keys are community identifiers and values are lists of nodes representing detected communities.
    - ground_truth: A dictionary where keys are community identifiers and values are lists of nodes representing ground-truth communities.

    Returns:
    - NMI: Normalized Mutual Information
    """
    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)
    
    # Create label arrays for ARI and NMI calculations
    node_list = list(G.nodes())
    partition_labels = [None] * len(node_list)
    ground_truth_labels = [None] * len(node_list)

    # Assign labels according to partition
    for label, nodes in partition.items():
        for node in nodes:
            partition_labels[node_list.index(node)] = label

    # Assign labels according to ground truth
    for label, nodes in ground_truth.items():
        for node in nodes:
            ground_truth_labels[node_list.index(node)] = label

    # Compute ARI and NMI
    nmi = normalized_mutual_info_score(ground_truth_labels, partition_labels)

    return nmi


def entropy(labels):
    """Calculate the entropy of a labeling."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


def mutual_information(partition_labels, ground_truth_labels):
    """Calculate the mutual information between two labelings."""
    contingency_table = np.histogram2d(partition_labels, ground_truth_labels,
                                       bins=(np.unique(partition_labels).size, np.unique(ground_truth_labels).size))[0]
    total = contingency_table.sum()
    PI = contingency_table / total
    marginal_i = PI.sum(axis=1).reshape(-1, 1)
    marginal_j = PI.sum(axis=0).reshape(1, -1)
    MI = np.sum(PI * (np.log(PI + 1e-10) - np.log(marginal_i) - np.log(marginal_j)))
    return MI


def compute_metrics(G, partition, ground_truth):
    """Compute VI, Modified Purity, Modified ARI, and Modified NMI."""
    node_list = list(G.nodes())
    partition_labels = [None] * len(node_list)
    ground_truth_labels = [None] * len(node_list)

    for label, nodes in partition.items():
        for node in nodes:
            partition_labels[node_list.index(node)] = label

    for label, nodes in ground_truth.items():
        for node in nodes:
            ground_truth_labels[node_list.index(node)] = label

    # Entropy and Mutual Information
    H_X = entropy(partition_labels)
    H_Y = entropy(ground_truth_labels)
    I_XY = mutual_information(partition_labels, ground_truth_labels)

    # Variation of Information
    VI = H_X + H_Y - 2 * I_XY

    # Modified Purity
    max_overlap = max(np.histogram2d(partition_labels, ground_truth_labels,
                                     bins=(np.unique(partition_labels).size, np.unique(ground_truth_labels).size))[
                          0].max(axis=0))
    modified_purity = max_overlap / len(G.nodes())

    # Modified ARI
    TP_plus_FP = comb(np.bincount(partition_labels), 2).sum()
    TP_plus_FN = comb(np.bincount(ground_truth_labels), 2).sum()
    TP = sum(comb(np.histogram2d(partition_labels, ground_truth_labels,
                                 bins=(np.unique(partition_labels).size, np.unique(ground_truth_labels).size))[0],
                  2).sum())
    FP = TP_plus_FP - TP
    FN = TP_plus_FN - TP
    TN = comb(len(G.nodes()), 2) - TP - FP - FN
    modified_ari = (TP + TN) / (TP + FP + FN + TN)

    # Modified NMI
    modified_nmi = 2 * I_XY / (H_X + H_Y)

    return VI, modified_purity, modified_ari, modified_nmi


def edit_distance_community(partition1, partition2):
    """
    Compute the Edit Distance between two community structures.
    Each move of a node from one community to another counts as one edit.

    Parameters:
    - partition1: A dictionary where keys are community identifiers and values are lists of nodes (first partition).
    - partition2: A dictionary where keys are community identifiers and values are lists of nodes (second partition).

    Returns:
    - edit_distance: The minimum number of node moves required to transform partition1 into partition2.
    """
    partition1=convert_to_nx_partition(partition1)
    partition2=convert_to_nx_partition(partition2)
    
    # Create node to community maps
    node_to_community1 = {node: cid for cid, nodes in partition1.items() for node in nodes}
    node_to_community2 = {node: cid for cid, nodes in partition2.items() for node in nodes}

    # Count the number of nodes that need to be moved
    moves = 0
    for node, community1 in node_to_community1.items():
        community2 = node_to_community2.get(node)
        if community1 != community2:
            moves += 1

    return moves
def main_non_overlapping_metrics_with_groundtruth(G,partition,ground_truth):
    purity,inverse_purity=compute_purity(partition, ground_truth)
    # vi, mod_purity, mod_ari, mod_nmi = compute_metrics(G, partition, ground_truth)
    dict={
        "Purity":purity,
        "F-measure":compute_f_measure(purity, inverse_purity),
        "f1": evaluation.f1(partition, ground_truth).score,

        # "RI":compute_rand_index(partition, ground_truth),
        "AMI":evaluation.adjusted_mutual_information(partition,ground_truth).score,
        "RI":evaluation.rand_index(partition,ground_truth).score,
        # "ARI":compute_ari(G, partition, ground_truth),
        "ARI":evaluation.adjusted_rand_index(partition,ground_truth).score,
        # "NMI":compute_nmi(G, partition, ground_truth),
        "NMI":evaluation.normalized_mutual_information(partition,ground_truth).score,
        # "Variation of Information (VI)": vi,
        "Variation of Information (VI)": evaluation.variation_of_information(partition,ground_truth).score,

        # "Modified Purity": mod_purity,
        # "Modified ARI":mod_ari,
        # "Modified NMI": mod_nmi,
        "Edit Distance": edit_distance_community(partition, ground_truth),
        }

    # dict={
    #     "Purity":purity,
    #     "F-measure":compute_f_measure(purity, inverse_purity),
    #     "f1": evaluation.f1(partition, ground_truth).score,
    #     "Edit Distance": edit_distance_community(partition, ground_truth),
    #     }

    return dict