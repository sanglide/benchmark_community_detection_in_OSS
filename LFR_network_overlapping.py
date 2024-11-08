import platform
import sys
import networkx as nx
import os
import pickle
from cdlib import algorithms
from metricsOverlappingWithGroundTruth import main_overlapping_metrics_with_groundtruth
from cdlib import NodeClustering
import random
import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from math import comb
import numpy as np
import csv
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"


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

def draw_LFR_graph(G,communities,filename):
    file_path=f'{prefix}temp/lfr-graphs/pic/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

     # Create a color map
    num_communities = len(communities)
    color_map = {}

    for i, community in enumerate(communities):
        color = (random.random(), random.random(), random.random())  # Random color
        for node in community:
            color_map[node] = color

    node_colors = [color_map.get(node, (0.5, 0.5, 0.5)) for node in G.nodes()]  # Default to gray if not in any community

    # Draw the graph
    fig,ax=plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, node_color=node_colors, node_size=100, with_labels=False, edge_color='gray', alpha=0.6)

    # Add a colorbar legend
    # sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=num_communities-1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm,ax=ax)
    # cbar.set_label('Communities')

    # Save the figure
    # plt.title(f'{filename} Graph with Communities')``
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{file_path}{filename}-net.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph saved as {f'{file_path}{filename}-net.png'}")
    

def no_console_output(func):
    def wrapper(*args, **kwargs):
        # 保存当前的标准输出
        original_stdout = sys.stdout
        # 打开一个空文件进行重定向
        with open(r'nul', 'w') as devnull:
            sys.stdout = devnull
            try:
                # 调用目标函数
                result = func(*args, **kwargs)
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout
        return result
    return wrapper
 


def read_network(index):
    input_path_c=f'{prefix}data/lfr_network/{index}-community.dat'
    input_path_n=f'{prefix}data/lfr_network/{index}-network.dat'

    with open(input_path_c, 'r') as file:
        data_c = file.readlines()

    data_c = [line.strip().split() for line in data_c]
    dict_community={}
    for node_c in data_c:
        node_index=node_c[0]
        for c in node_c[1:]:
            if c in dict_community.keys():
                dict_community[c].append(node_index)
            else:
                dict_community[c]=[node_index]
    community_list=list(dict_community.values())

    print(f'**  read ground-truth communities has done!  **')

    with open(input_path_n, 'r') as file:
        data_n = file.readlines()

    data_n = [line.strip().split() for line in data_n[1:]]
    
    graph_dict={}
    for edge in data_n:
        node1,node2,weight=edge
        if node1 in graph_dict.keys():
            graph_dict[node1][node2]={"weight":weight}
        else:
            graph_dict[node1]={node2:{"weight":weight}}
    G=nx.Graph(graph_dict)

    print(f'**  read ground-truth graph has done!  **')

    statistic_LFR_graph_percolation_transition2(index,G)
    statistic_Q(f'{prefix}temp/statistics/q-lfr-overlapping.csv',index,G)

    draw_LFR_graph(G,community_list,f'{index}-LFR-overlapping')

    return community_list,G

def save_communities(community_obj, name,index):
    """
    使用pickle模块保存NodeClustering对象到文件。
    """
    if not os.path.exists(f'{prefix}temp/lfr-communities-overlapping'):
        os.makedirs(f'{prefix}temp/lfr-communities-overlapping')
    with open(f'{prefix}temp/lfr-communities-overlapping/' +str(index)+"-"+ name + '.pkl', 'wb') as f:
        pickle.dump(community_obj, f)

def find_first_occurrence(d, value):
    return next((k for k, v in d.items() if v == value), None)

@no_console_output
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

def generate_communities(G,index):
    # K-Clique算法
    print("Running K-Clique...")
    communities_kclique = algorithms.kclique(G, k=4)
    draw_LFR_graph(G,communities_kclique.communities,f'{index}-Kclique')
    save_communities(communities_kclique, 'Kclique',index)
    print("K-Clique is done.")

    # SLPA算法
    print("Running SLPA...")
    communities_slpa = algorithms.slpa(G)
    save_communities(communities_slpa, 'SLPA',index)
    draw_LFR_graph(G,communities_slpa.communities,f'{index}-SLPA')
    print("SLPA is done.")

    # DEMON算法
    print("Running DEMON...")
    communities_demon = algorithms.demon(G, min_com_size=3, epsilon=0.25)
    save_communities(communities_demon, 'DEMON',index)
    draw_LFR_graph(G,communities_demon.communities,f'{index}-DEMON')
    print("DEMON is done.")

    # LEMON算法
    print("Running LEMON...")

    communities_lemon=get_all_communities_by_lemon(G)
    save_communities(communities_lemon, 'LEMON',index)
    draw_LFR_graph(G,communities_lemon.communities,f'{index}-LEMON')
    print("LEMON is done.")

    print("All the communities are detected and saved.")

def metrics_lfr_network_overlapping(index):

    ground_truth,G=read_network(index)
    ground_truth_NC=NodeClustering(ground_truth,G)

    generate_communities(G,index)

    print(f'**  communities detection has done!  **')

    dict={}
    overlapping_alg = ["Kclique", "SLPA", "LEMON", "DEMON"]
    for alg in overlapping_alg:
        with open(f'{prefix}temp/lfr-communities-overlapping/' +str(index)+"-"+ alg + '.pkl', 'rb') as f:
            partition = pickle.load(f)
        if partition == "false":
            print(f'the community detection algo ** {alg} ** has failed!')
            continue
        dict[f'{alg}_groundtruth'] = main_overlapping_metrics_with_groundtruth(G, partition,ground_truth_NC)
    json_data = json.dumps(dict)

    # 将JSON字符串写入到文件
    with open(f'{prefix}outputs/lfr-ground-truth-metrics/{index}-lfr-result-overlapping.json', 'w') as f:
        f.write(json_data)


def main_LFR_network_overlapping(index_range):
    for index in range(index_range):
        metrics_lfr_network_overlapping(index)

def draw_csv_pic(index_range):
    input_path=f'{prefix}outputs/lfr-ground-truth-metrics/'
    output_csv_path=f"{prefix}outputs/graph-metrics-figcsv/"

    overlapping_alg = ["Kclique", "SLPA", "LEMON", "DEMON"]

    mean_list=[]
    df_list=[]
    df_names=[]
    for alg in overlapping_alg:
        dict_list=[]
        for i in range(index_range):
            filename=f'{input_path}{i}-lfr-result-overlapping.json'
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if bool(data) and (f'{alg}_groundtruth' in data.keys()):
                dict_list.append(data[f'{alg}_groundtruth'])

        df_n = pd.DataFrame(dict_list)
        column_means = df_n.mean()
        column_means.name=alg
        mean_list.append(column_means)
        df_list.append(df_n)
        df_names.append(alg)
        df_n.to_csv(f'{output_csv_path}/lfr_{alg}_overlapping_metrics.csv')
    dff=pd.DataFrame({s.name: s for s in mean_list})
    dff.to_csv(f'{output_csv_path}lfr_overlapping-algo-metric.csv')

    for key in dict_list[0].keys():
        merged_df = pd.DataFrame()
        # Loop through the DataFrames and extract the 'a' column
        for df, name in zip(df_list, df_names):
            merged_df[name] = df[key]

        # Display the merged DataFrame
        merged_df.to_csv(f'{output_csv_path}lfr_overlapping-algo-{key}-metrics.csv')

def statistic_LFR_graph_percolation_transition2(index,G):
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_nodes < 2:
        print(f'Graph only has two nodes')
        return
    p = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    text_to_append=f"\n{index}-LFR-graph \n Probability p of edge connection: {p}\n"
    
    for k in range(3,7):
        pc = 1 / (((k-1)*num_nodes)**(1/(k-1)))
        text_to_append+=f"Critical probability p_c({k}): {pc}\n"
    
    with open('count_LFR_graph_percolation_transition.txt', 'a', encoding='utf-8') as file:
        file.write(text_to_append)

def statistic_LFR_graph_percolation_transition(index,G):
    sizes = []
    edges=[]

    for edge in G.edges():
        edges.append((edge[0],edge[1], float(G[edge[0]][edge[1]]['weight'])))

    max_weight = max(weight for _, _, weight in edges)
    p_values = np.linspace(0, max_weight, 100)

    for p in p_values:
        # Create a subgraph based on edge weights
        # Keep edges with weight greater than p
        edges_to_keep = [(u, v) for u, v, weight in edges if weight >= p]
        H = nx.Graph()
        H.add_edges_from(edges_to_keep)

        # Find the largest connected component
        if len(H.nodes()) > 0:
            largest_cc = max(nx.connected_components(H), key=len)
            sizes.append(len(largest_cc))
        else:
            sizes.append(0)  # No edges kept, size is 0
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, sizes, marker='o')
    plt.title('Percolation Transition on Weighted Graph')
    plt.xlabel('Edge Weight Threshold (p)')
    plt.ylabel('Size of Largest Connected Component')
    plt.axvline(x=2.5, color='r', linestyle='--', label='Example Threshold')
    plt.legend()
    plt.grid()
    plt.savefig(f'{prefix}temp/lfr-graphs/pic/{index}-percolation-transition.png')
    plt.close()

    



index_range=10
main_LFR_network_overlapping(index_range)
draw_csv_pic(index_range)