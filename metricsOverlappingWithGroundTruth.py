from sklearn.metrics import normalized_mutual_info_score
from cdlib import evaluation, algorithms


'''
----------------------------------------------------
3.2 Ground-Truth-Based Validation Measures for Overlapping Community Structure

(1) ONMI : compute_onmi(G,partition, ground_truth)
(2) Omega Index : compute_omega_index(G,partition, ground_truth)
(3) Generalized External Index : compute_generalized_external_index(G,partition, ground_truth)
(4) Fuzzy Rand Index : compute_fuzzy_rand_index(G,partition, ground_truth)
(5) F1-score, sensitivity, specificity and accuracy

'''

def convert_to_nx_partition(node_clustering):
    partition = {}
    for community_id, community in enumerate(node_clustering.communities):
        partition[community_id]=community
    return partition

def compute_onmi(partition, ground_truth):
    """
    Compute the Overlapping Normalized Mutual Information (ONMI) for two community structures.
    """
    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)

    # Flatten the community structures and assign unique labels to each community
    partition_labels = {node: i for i, community in enumerate(partition) for node in community}
    ground_truth_labels = {node: i for i, community in enumerate(ground_truth) for node in community}

    # Create label vectors based on the community assignments
    labels1 = [partition_labels.get(node, -1) for node in set(partition_labels) | set(ground_truth_labels)]
    labels2 = [ground_truth_labels.get(node, -1) for node in set(partition_labels) | set(ground_truth_labels)]

    return normalized_mutual_info_score(labels1, labels2)


def compute_omega_index(partition, ground_truth):
    """
    Compute the Omega Index for two community structures.
    """

    partition=convert_to_nx_partition(partition)
    ground_truth=convert_to_nx_partition(ground_truth)

    # Create sets of node pairs for each community
    def get_pairs(community):
        return set(frozenset((u, v)) for u in community for v in community if u != v)

    partition_pairs = [get_pairs(community) for community in partition]
    ground_truth_pairs = [get_pairs(community) for community in ground_truth]

    # Calculate the expected and observed overlaps
    observed = sum(len(c1 & c2) for c1 in partition_pairs for c2 in ground_truth_pairs)
    expected = sum(len(c1) * len(c2) for c1 in partition_pairs for c2 in ground_truth_pairs) / (
                len(partition) * len(ground_truth))

    return (observed - expected) / (len(partition) * len(ground_truth) - expected)

def main_overlapping_metrics_with_groundtruth(G,partition,ground_truth):
    # f1_score, sensitivity, specificity, accuracy=compute_f1_sensitivity_specificity_accuracy(G, partition, ground_truth)
    # dict={
    #     "ONMI" : compute_onmi(partition, ground_truth),
    #     "OI" : compute_omega_index(G,partition, ground_truth),
    #     "GEI": compute_generalized_external_index(G,partition, ground_truth),
    #     "FRI":compute_fuzzy_rand_index(G,partition, ground_truth),
    #     "F1-score":f1_score,
    #     "sensitivity": sensitivity,
    #     "specificity":specificity,
    #     "accuracy": accuracy,
    #     }
    dict = {
        "f1": evaluation.f1(partition, ground_truth).score,
        "OI": evaluation.overlapping_normalized_mutual_information_LFK(partition,ground_truth).score,
        # "geometric_accuracy":evaluation.geometric_accuracy(partition,ground_truth).score,
        # "quality":evaluation.overlap_quality(partition,ground_truth).score,
        "NMI_MGH":evaluation.overlapping_normalized_mutual_information_MGH(partition,ground_truth).score,
        "NMI__LFK":evaluation.overlapping_normalized_mutual_information_LFK(partition,ground_truth).score,
        # "ONMI" : compute_onmi(partition, ground_truth),
    }


    return dict