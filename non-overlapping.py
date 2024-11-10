# implementation of non-overlapping community detection algorithms
import argparse
import pickle

import networkx as nx
import os
from cdlib import algorithms

import platform
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()


G = nx.read_graphml(f'{prefix}temp/graphs/{args.proj.replace("/","@")}-graph.graphml')

print("Running Girvan-Newman...")
communities_gn = algorithms.girvan_newman(G, level=3)
print("Girvan-Newman is done.")

print("Running walktrap...")
communities_walktrap=algorithms.walktrap(G)
print("Walktrap is done.")

print("Running Louvain...")
communities_louvain = algorithms.louvain(G)
print("Louvain is done.")

print("Running CNM...")
communities_cnm = algorithms.greedy_modularity(G)
print("CNM is done.")

print("Running LPA...")
communities_lpa = algorithms.label_propagation(G)
print("LPA is done.")

print("Running Infomap...")
communities_infomap = algorithms.infomap(G)
print("Infomap is done.")

print("Running Eigenvector...")
communities_eig = algorithms.eigenvector(G)
print("Eigenvector is done.")

avg_communities = round((len(communities_gn.communities)+len(communities_louvain.communities)+len(communities_lpa.communities)+
                         len(communities_infomap.communities)+len(communities_eig.communities))/5)
# print("avg_com:", avg_communities)

print("Running AGDL...")
communities_agdl = algorithms.agdl(G, number_communities=avg_communities, kc=10)
print("AGDL is done.")

print("Running EM...")
communities_em = algorithms.em(G, k=avg_communities)
print("EM is done.")

print("All the communities are detected.")


def save_communities(community_obj, name,proj):
    if not os.path.exists(f'{prefix}temp/communities'):
        os.makedirs(f'{prefix}temp/communities')
    with open(f'{prefix}temp/communities/' +proj.replace("/","@")+"-"+ name + '.pkl', 'wb') as f:
        pickle.dump(community_obj, f)


save_communities(communities_gn, 'Girvan-Newman',args.proj)
save_communities(communities_cnm, 'CNM',args.proj)
save_communities(communities_louvain, 'Louvain',args.proj)
save_communities(communities_lpa, 'LPA',args.proj)
save_communities(communities_infomap, 'Infomap',args.proj)
save_communities(communities_agdl, 'AGDL',args.proj)
save_communities(communities_eig, 'Eigenvector',args.proj)
save_communities(communities_em, 'EM',args.proj)
save_communities(communities_walktrap,'Walktrap'.args.proj)
