# implementation of overlapping community detection algorithms
import argparse
import random
import pickle
import os
from cdlib import NodeClustering
from collections import Counter
import networkx as nx
from cdlib import algorithms, viz

import platform
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"
def find_first_occurrence(d, value):
    return next((k for k, v in d.items() if v == value), None)
def get_all_communities_by_lemon(G):
    node_dict={}
    communities=[]
    for node in G.nodes():
        node_dict[node]=0
    prefix_count=len(G.nodes())
    count=0
    while (0 in node_dict.values()) and prefix_count!=count:
        prefix_count=count
        node_seed=find_first_occurrence(node_dict,0)
        coms = algorithms.lemon(G,[node_seed])
        communities.append(coms.communities[0])
        for node in coms.communities[0]:
            node_dict[node]=1
        node_dict[node_seed]=1
        counts = Counter(node_dict.values())
        count = counts[0]
        text_to_append = f"{count}\n{node_seed}\n{communities}\n{node_dict}\n---------------\n"
        with open('count_lemon_wajue.txt', 'a', encoding='utf-8') as file:
            file.write(text_to_append)
    c_N_c=NodeClustering(communities,G)
    return c_N_c


parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()

# 读取图
G = nx.read_graphml(f'{prefix}temp/graphs/{args.proj.replace("/","@")}-graph.graphml')
# G = nx.karate_club_graph()

# K-Clique算法
print("Running K-Clique...")
communities_kclique = algorithms.kclique(G, k=6)
print("K-Clique is done.")

# SLPA算法
print("Running SLPA...")
communities_slpa = algorithms.slpa(G)
print("SLPA is done.")

# DEMON算法
print("Running DEMON...")
communities_demon = algorithms.demon(G, min_com_size=3, epsilon=0.25)
print("DEMON is done.")

avg_communities = round((len(communities_kclique.communities)+len(communities_slpa.communities)+
                         +len(communities_demon.communities))/3)

# LEMON算法
all_nodes = list(G.nodes())
random_seeds = random.sample(all_nodes, avg_communities)
print("Running LEMON...")
communities_lemon = get_all_communities_by_lemon(G)
print("LEMON is done.")

# avg_communities = round((len(communities_kclique.communities)+len(communities_slpa.communities)+
#                          len(communities_lemon.communities)+len(communities_demon.communities))/4)
#
# CONGA算法
# print("Running CONGA...")
# communities_conga = algorithms.conga(G)
# print("CONGA is done.")


# Save communities
def save_communities(community_obj, name,proj):
    if not os.path.exists(f'{prefix}temp/communities'):
        os.makedirs(f'{prefix}temp/communities')
    with open(f'{prefix}temp/communities/' +proj.replace("/","@")+"-"+ name + '.pkl', 'wb') as f:
        pickle.dump(community_obj, f)


save_communities(communities_kclique, 'Kclique',args.proj)
save_communities(communities_slpa, 'SLPA',args.proj)
# save_communities(communities_conga, 'CONGA',args.proj)
save_communities(communities_lemon, 'LEMON',args.proj)
save_communities(communities_demon, 'DEMON',args.proj)

print("All the communities are detected and saved.")
