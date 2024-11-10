import itertools
import json
import os
import pickle
import networkx as nx
from cdlib import NodeClustering
from networkx.generators.community import LFR_benchmark_graph
import argparse
import platform
import powerlaw

from metricsNonOverlappingWithGroundTruth import main_non_overlapping_metrics_with_groundtruth
from metricsOverlappingWithGroundTruth import main_overlapping_metrics_with_groundtruth

system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()

def load_communities(file_path):
    if not os.path.exists(file_path):
        return "false"
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_to_nx_partition(G):
    partition = {}
    for node_id in G:
        community_list=G.nodes[node_id]["community"]
        for community in community_list:
            print(community)
            if community not in partition.keys():
                partition[community] = []
            partition[community].append(node_id)
    return partition

def convert_to_cdlib_partition(G):
    partition = {}
    for node_id in G:
        community_list = G.nodes[node_id]["community"]
        for community in community_list:
            if community not in partition.keys():
                partition[community] = []
            partition[community].append(node_id)
    communities=[]
    for community_id in partition.keys():
        communities.append(partition[community_id])
    coms = NodeClustering(communities, graph=G)
    return coms


def calculate_graph_attributes(G):
    n = len(G.nodes())
    # Calculate tau1
    degree_sequence = [d for n, d in G.degree()]
    fit_degree = powerlaw.Fit(degree_sequence)
    tau1 = fit_degree.alpha  # Power law exponent for degree distribution

    # Calculate tau2
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    community_sizes = [len(c) for c in communities]
    fit_community = powerlaw.Fit(community_sizes)
    tau2 = fit_community.alpha  # Power law exponent for community size distribution

    # Calculate mu
    inter_community_edges = 0
    total_edges = G.number_of_edges()

    for i, community in enumerate(communities):
        for j in range(i + 1, len(communities)):
            inter_community_edges += len(set(community).intersection(communities[j]))

    mu = inter_community_edges / total_edges if total_edges > 0 else 0  # Fraction of inter-community edges

    min_degree = min(degree_sequence) if degree_sequence else 0
    average_degree = sum(degree_sequence) / len(degree_sequence) if degree_sequence else 0

    return n,tau1, tau2, mu,min_degree,average_degree

def main_metric_with_ground_truth(proj):
    if not os.path.exists(f'{prefix}outputs/ground-truth-metrics/'):
        os.mkdir(f'{prefix}outputs/ground-truth-metrics/')

    overlapping_alg = ["Kclique", "SLPA", "LEMON", "DEMON"]
    non_overlapping_alg = ['Girvan-Newman', 'Louvain', 'LPA', 'Infomap', 'AGDL', 'Eigenvector', 'EM','CNM']

    dict = {}
    G = nx.read_graphml(f'{prefix}temp/graphs/{proj.replace("/", "@")}-graph.graphml')

    for o_l1,o_l2 in list(itertools.product(overlapping_alg, repeat=2)):
        partition = load_communities(f'{prefix}temp/communities/{proj.replace("/", "@")}-{o_l1}.pkl')
        ground_truth = load_communities(f'{prefix}temp/communities/{proj.replace("/", "@")}-{o_l2}.pkl')
        if partition == "false":
            print(f'the community detection algo ** {o_l1} {o_l2} ** in ** {proj} ** has failed!')
            continue
        dict[f'{o_l1}_{o_l2}'] = main_overlapping_metrics_with_groundtruth(G,partition,ground_truth)
    for n_l1,n_l2 in list(itertools.product(non_overlapping_alg, repeat=2)):
        partition = load_communities(f'{prefix}temp/communities/{proj.replace("/", "@")}-{n_l1}.pkl')
        ground_truth = load_communities(f'{prefix}temp/communities/{proj.replace("/", "@")}-{n_l2}.pkl')
        if partition == "false":
            print(f'the community detection algo ** {n_l1} {n_l2} ** in ** {proj} ** has failed!')
            continue
        dict[f'{n_l1}_{n_l2}'] = main_non_overlapping_metrics_with_groundtruth(G, partition,ground_truth)


    json_data = json.dumps(dict)

    # 将JSON字符串写入到文件
    with open(f'{prefix}outputs/ground-truth-metrics/{args.proj.replace("/", "@")}.json', 'w') as f:
        f.write(json_data)


main_metric_with_ground_truth(args.proj)
