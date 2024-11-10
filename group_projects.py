import networkx as nx
import powerlaw
import numpy as np
import platform
import argparse
import os
import csv
import pickle
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="project name")
args = parser.parse_args()

def convert_node_clustering_to_partition_dict(node_clustering):
    dict={}
    for i,community in enumerate(node_clustering.communities):
        dict[i]=community
    return dict

def load_communities(file_path):

    if not os.path.exists(file_path):
        return "false"
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_mu(G, partition):
    mu_values = {}
    
    # Create a mapping of node to its community
    node_to_community = {}
    for community, nodes in partition.items():
        for node in nodes:
            node_to_community[node] = community
    # Calculate mu for each node
    # print(node_to_community)
    for node in node_to_community.keys():
        k_in = 0
        k_out = 0
        
        # Get the community of the current node
        current_community = node_to_community[node]
        
        # Iterate through neighbors to calculate k_in and k_out
        for neighbor in G.neighbors(node):
            
            if neighbor in node_to_community.keys() and node_to_community[neighbor] == current_community:
                # print(f'neighbor: {neighbor}, node: {node} , community-neighbor : {node_to_community[neighbor]} , current community : {current_community}')
                k_in += 1  # Neighbor is in the same community
            else:
                k_out += 1  # Neighbor is in a different community

        # Calculate mu if k_in + k_out > 0 to avoid division by zero
        # print(f'k in : {k_in} , k_out : {k_out}')
        if (k_in + k_out) > 0:
            mu = k_out / (k_in + k_out)
        else:
            mu = 0.0  # Handle case where there are no neighbors

        mu_values[node] = mu

    return mu_values

def get_properties_of_proj(G,partition):
    # Analyze node degree distribution
    degrees = [d for n, d in G.degree()]
    degree_results = powerlaw.Fit(degrees)
    degree_alpha = degree_results.power_law.alpha
    degree_ks = degree_results.power_law.KS()
    degree_is_power_law = degree_results.distribution_compare('power_law', 'exponential')[0] > 0

    print(f'Average Node Degree <k> : {np.mean(degrees)}')
    print(f"Node Degree Distribution: alpha={degree_alpha}, KS={degree_ks}, is_power_law={degree_is_power_law}")

    # Analyze community size distribution
    community_sizes = [len(partition[community]) for community in partition]
    community_results = powerlaw.Fit(community_sizes)
    community_alpha = community_results.power_law.alpha
    community_ks = community_results.power_law.KS()
    community_is_power_law = community_results.distribution_compare('power_law', 'exponential')[0] > 0



    print(f"Community Size Distribution: alpha={community_alpha}, KS={community_ks}, is_power_law={community_is_power_law}")

    mu=compute_mu(G,partition)

    return {
        'average degree': np.mean(degrees),
        'node_degree_alpha': degree_alpha,
        'node_degree_ks': degree_ks,
        'node_degree_is_power_law': degree_is_power_law,
        'community_size_alpha': community_alpha,
        'community_size_ks': community_ks,
        'community_size_is_power_law': community_is_power_law,
        "mu":mu,
    }


def get_properties(proj):
    overlapping_alg=["Kclique","SLPA","LEMON","DEMON",'CONGA']
    non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','AGDL','Eigenvector','EM','CNM']

    
    if not os.path.exists(f'{prefix}temp/statistics'):
        os.makedirs(f'{prefix}temp/statistics')
    file_path=f'{prefix}temp/statistics/communities-properties.csv'

    file_path1=f'{prefix}temp/statistics/communities-properties_graph.csv'

    # if os.path.isfile(file_path):
    #     os.remove(file_path)

    G = nx.read_graphml(f'{prefix}temp/graphs/{proj.replace("/", "@")}-graph.graphml')

    degrees = [d for n, d in G.degree()]
    degree_results = powerlaw.Fit(degrees)
    degree_alpha = degree_results.power_law.alpha
    degree_ks = degree_results.power_law.KS()
    degree_is_power_law = degree_results.distribution_compare('power_law', 'exponential')[0] > 0
    dict_node={
        "Average Node Degree": np.mean(degrees),
        "Power Law Exponent": degree_alpha,
        "KL Divergence": degree_ks,
        "Fits Power Law": degree_is_power_law,
        'proj':proj
    }
    with open(file_path1, mode='a', newline='') as csvfile:
        fieldnames = dict_node.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.isfile(file_path1):
            writer.writeheader()
        writer.writerow(dict_node)

    for o_l in  overlapping_alg:
        temp_dict={}
        partition=load_communities(f'{prefix}temp/communities/{proj.replace("/","@")}-{o_l}.pkl')
        if partition=="false":
            print(f'the community detection algo ** {o_l} ** in ** {proj} ** has failed!')
            continue
        temp_dict=get_properties_of_proj(G,convert_node_clustering_to_partition_dict(partition))
        temp_dict["proj"]=proj
        temp_dict['algo']=o_l
        
        with open(file_path, mode='a', newline='') as csvfile:
            fieldnames = temp_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.isfile(file_path):
                writer.writeheader()
            writer.writerow(temp_dict)
    for n_l in non_overlapping_alg:
        temp_dict={}
        partition=load_communities(f'{prefix}temp/communities/{proj.replace("/","@")}-{n_l}.pkl')
        if partition=="false":
            print(f'the community detection algo ** {n_l} ** in ** {proj} ** has failed!')
            continue
        temp_dict=get_properties_of_proj(G,convert_node_clustering_to_partition_dict(partition))
        temp_dict["proj"]=proj
        temp_dict['algo']=n_l
        
        with open(file_path, mode='a', newline='') as csvfile:
            fieldnames = temp_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.isfile(file_path):
                writer.writeheader()
            writer.writerow(temp_dict)

    
    

    
get_properties(args.proj)