import platform
import argparse
import os
import networkx as nx
from collections import Counter
import numpy as np
import json

system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()


def compute_network_robustness(G):
    original_size = len(G)
    node_connectivity = nx.node_connectivity(G)
    edge_connectivity = nx.edge_connectivity(G)
    
    # Simulate random node removals
    robustness_score = 0
    num_simulations = 100
    for _ in range(num_simulations):
        H = G.copy()
        nodes_to_remove = int(0.1 * original_size)  # Remove 10% of nodes
        for _ in range(nodes_to_remove):
            if len(H) > 1:
                node = np.random.choice(list(H.nodes()))
                H.remove_node(node)
        largest_cc = max(nx.connected_components(H), key=len)
        robustness_score += len(largest_cc) / original_size
    
    robustness_score /= num_simulations
    
    return {
        'node_connectivity': node_connectivity,
        'edge_connectivity': edge_connectivity,
        'robustness_score': robustness_score
    }

def get_all_metrics(G):
    metrics = {}
    
    # Number of nodes
    metrics['num_nodes'] = nx.number_of_nodes(G)
    
    # Node degrees
    degrees = [d for n, d in G.degree()]
    metrics['max_degree'] = max(degrees)
    metrics['min_degree'] = min(degrees)
    metrics['avg_degree'] = sum(degrees) / len(degrees) if len(degrees)!=0 else 0
    
    # Cliques
    cliques = list(nx.find_cliques(G))
    clique_sizes = [len(c) for c in cliques]
    metrics['num_cliques'] = len(cliques)
    metrics['max_clique_size'] = max(clique_sizes)
    metrics['min_clique_size'] = min(clique_sizes)
    metrics['avg_clique_size'] = sum(clique_sizes) / len(clique_sizes)
    metrics['clique_size_distribution'] = dict(Counter(clique_sizes))
    
    # Local clustering coefficients
    metrics['avg_clustering_coefficients'] = nx.average_clustering(G)
    
    # Centrality measures
    metrics['avg_closeness_centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
    metrics['avg_betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
    metrics['avg_eigenvector_centrality'] = np.mean(list(nx.eigenvector_centrality(G).values()))

    # diameter
    metrics['diameter']=nx.diameter(G)

    # average path length
    metrics['average_path_length']=nx.average_shortest_path_length(G)

    # density
    metrics['density']=nx.density(G)

    # network robustness
    metrics['network_robustness']=compute_network_robustness(G)

    # edge_weight
    weights = [d['weight'] for (u, v, d) in G.edges(data=True) if 'weight' in d]
    if not weights:
        return None, None, None, None
    
    metrics['max_weight'] = max(weights)
    metrics['min_weight'] = min(weights)
    metrics['avg_weight'] = np.mean(weights)
    
    return metrics


def main_metrics(proj):
    if not os.path.exists(f'{prefix}outputs/graph-metrics/'):
        os.mkdir(f'{prefix}outputs/graph-metrics/')

    
    G = nx.read_graphml(f'{prefix}temp/graphs/{proj.replace("/", "@")}-graph.graphml')
    metrics=get_all_metrics(G)

    json_data = json.dumps(metrics)

    # 将JSON字符串写入到文件
    with open(f'{prefix}outputs/graph-metrics/{args.proj.replace("/", "@")}.json', 'w') as f:
        f.write(json_data)
   

main_metrics(args.proj)