import os.path
import argparse
import csv
import pandas as pd
import networkx as nx

import platform
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()

df = issue_pr_df=pd.read_csv(f"{prefix}data/{args.proj.replace('/', '@')}.csv")

# 创建无向赋权图
G = nx.Graph()
DG = nx.DiGraph()
for _, row in df.iterrows():
    create_user = row['createUser']
    comment_users = eval(row['commentsUser'])
    if create_user:
        if not G.has_node(create_user):
            G.add_node(create_user)
        if not DG.has_node(create_user):
            DG.add_node(create_user)
        for comment_user in comment_users:
            if comment_user == create_user:
                continue
            if not G.has_node(comment_user):
                G.add_node(comment_user)
            if not DG.has_node(comment_user):
                DG.add_node(comment_user)
            if G.has_edge(create_user, comment_user):
                G[create_user][comment_user]['weight'] += 1
            else:
                G.add_edge(create_user, comment_user, weight=1)
            if DG.has_edge(create_user, comment_user):
                DG[create_user][comment_user]['weight'] += 1
            else:
                DG.add_edge(create_user, comment_user, weight=1)

# 移除孤立点
isolated = list(nx.isolates(G))
# print('isolated', isolated)
if isolated:
    G.remove_nodes_from(isolated)

def statistic_graph_percolation_transition(name,G):
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_nodes < 2:
        print(f'Graph only has two nodes')
        return
    p = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    text_to_append=f"\n{name} \n Probability p of edge connection: {p}\n"
    
    k_value=[]
    for k in range(3,7):
        pc = 1 / (((k-1)*num_nodes)**(1/(k-1)))
        k_value.append(pc)
        text_to_append+=f"Critical probability p_c({k}): {pc}\n"
    for index, item in enumerate(k_value):
        if item > p:
            text_to_append+=f'Proper k is : {index+3}\n'
            break
    
    text_to_append+=f'n : {len(G.nodes())}\n'

    with open('count_OSS_graph_percolation_transition.txt', 'a', encoding='utf-8') as file:
        file.write(text_to_append)


def count_complete_subgraphs(G):
    cliques = list(nx.find_cliques(G))
    num_cliques = len(cliques)
    return num_cliques

def statistic_Q(name,proj,G):
    n=len(G.nodes())
    m=count_complete_subgraphs(G)
    q_single=1-2/(m*(m-1)+2)-1/n
    q_pairs=1-1/(m*(m-1)+2)-2/n
    resolution_limit=True if q_single<=q_pairs else False

    dict_node={
        'proj':proj,
        'n':n,
        'm':m,
        'Q_single':q_single,
        'Q_pairs':q_pairs,
        'resolution limit':resolution_limit
    }
    with open(name, mode='a', newline='') as csvfile:
        fieldnames = dict_node.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.isfile(name):
            writer.writeheader()
        writer.writerow(dict_node)

# 保存
if not os.path.exists(f'{prefix}temp'):
    os.makedirs(f'{prefix}temp')
if not os.path.exists(f'{prefix}temp/graphs'):
    os.makedirs(f'{prefix}temp/graphs')
nx.write_graphml(G, f'{prefix}temp/graphs/{args.proj.replace("/","@")}-graph.graphml')
nx.write_graphml(DG, f'{prefix}temp/graphs/{args.proj.replace("/","@")}-directed-graph.graphml')

statistic_graph_percolation_transition(args.proj.replace("/","@"),G)
statistic_Q((f'{prefix}temp/statistics/q.csv'),args.proj,G)