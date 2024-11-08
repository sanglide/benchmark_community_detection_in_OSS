import argparse
import json
import os
import pickle
import networkx as nx
from metricsNonOverlapping import main_non_overlapping_metrics
from metricsOverlapping import main_overlapping_metrics

import platform
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()

def load_communities(file_path):
    """
    使用pickle加载保存的NodeClustering对象。
    """
    if not os.path.exists(file_path):
        return "false"
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def convert_to_communities(G, node_clustering):
    communities = node_clustering.communities
    
    # Check if communities form a partition
    all_nodes = set(G.nodes())
    community_nodes = set().union(*communities)
    
    if all_nodes != community_nodes:
        raise ValueError("Communities do not form a partition of the graph's nodes")
    
    return communities

def generate_metrics_for_community_detection(proj):
    columns=\
        ["proj","N_Modularity",
        "N_CommunityScore",
        "N_SPart",
        "N_Significance",
        "N_Permanence",
        "N_Surprise",
        "N_Communitude",
        "Modularity",
        "Flex"]
    
    if not os.path.exists(f'{prefix}outputs/metrics/'):
        os.mkdir(f'{prefix}outputs/metrics/')

    overlapping_alg=["Kclique","SLPA","LEMON","DEMON",'CONGA']
    non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','AGDL','Eigenvector','EM','CNM']

    
    dict={}
    G = nx.read_graphml(f'{prefix}temp/graphs/{proj.replace("/", "@")}-graph.graphml')

    # for o_l in  overlapping_alg:
    #     partition=load_communities(f'{prefix}temp/communities/{proj.replace("/","@")}-{o_l}.pkl')
    #     if partition=="false":
    #         print(f'the community detection algo ** {o_l} ** in ** {proj} ** has failed!')
    #         continue
    #     dict[o_l]=main_overlapping_metrics(G, partition)
    for n_l in non_overlapping_alg:
        partition=load_communities(f'{prefix}temp/communities/{proj.replace("/","@")}-{n_l}.pkl')
        if partition=="false":
            print(f'the community detection algo ** {n_l} ** in ** {proj} ** has failed!')
            continue
        dict[n_l]=main_non_overlapping_metrics(G, partition)

    # 将字典序列化为JSON格式的字符串
    json_data = json.dumps(dict)

    # 将JSON字符串写入到文件
    with open(f'{prefix}outputs/metrics/{args.proj.replace("/","@")}.json', 'w') as f:
        f.write(json_data)



generate_metrics_for_community_detection(args.proj)