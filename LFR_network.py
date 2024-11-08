import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from cdlib import algorithms
from cdlib import benchmark
import random
import numpy as np
from collections import Counter
import json
import itertools
from cdlib import NodeClustering
from metricsNonOverlappingWithGroundTruth import main_non_overlapping_metrics_with_groundtruth
from metricsOverlappingWithGroundTruth import main_overlapping_metrics_with_groundtruth
import platform
import csv
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

def draw_LFR_graph(G,communities,filename):
    file_path=f'{prefix}temp/lfr-graphs/pic/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

     # Create a color map
    num_communities = len(communities)
    color_map = plt.cm.get_cmap('tab20')

    # Assign colors to nodes based on their community
    node_colors = []
    for node in G.nodes():
        for i, comm in enumerate(communities):
            if node in comm:
                node_colors.append(color_map(i / num_communities))
                break

    # Draw the graph
    fig,ax=plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)

    
    communities_nodes_num=len(Counter([item for sublist in communities for item in sublist]))

    if len(G.nodes())==communities_nodes_num:

        nx.draw(G, pos, node_color=node_colors, node_size=100, with_labels=False, edge_color='gray', alpha=0.6)

        # Add a colorbar legend
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=num_communities-1))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=ax)
        # cbar.set_label('Communities')

        # Save the figure
        # plt.title(f'{filename} Graph with Communities')``
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{file_path}{filename}-net.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Graph saved as {f'{file_path}{filename}-net.png'}")
    else:
        print(f'G nodes :{len(G.nodes())}')
        print(f'communities nodes : {communities_nodes_num}')


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

def generate_lfr_benchmark(method):
    n = [1000,1000,1000,1000,1000,1000,
         1000,1000,1000,1000,1000]  # Number of nodes
    tau1 = [3,3.2,2.6,3.5,2.9,3.8,
            3,3.9,2,3.9,3.9]  # Node degree distribution power law exponent
    tau2 = [1.5,1.5,2,2.7,3.1,3.5,3,
            4,4.7,2.5,5.5,5.5]  # Community size distribution power law exponent
    mu = [0.1,0.2,0.3,0.1,0.2,0.5,
          0.6,0.7,0.8,0.9,1.0]  # Mixing parameter
    average_degree = [2.1,1.9,2.3,2.7,3,1.7,
                      2.5,2.9,4,7.7,7.7]
    min_community = [20,20,20,20,80,
                     20,20,20,80,20,20]

    # n=[100,1000,10000,100000]
    # tau1=3
    # tau2=2
    # # mu=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  
    # mu=0.1
    # min_degree=10
    # max_degree=50
    # min_community = 20

    for i in range(0, len(n)):
        print(f'the {i}-th LFR network')
        if not os.path.exists(f'{prefix}temp'):
            os.makedirs(f'{prefix}temp')
        if not os.path.exists(f'{prefix}temp/lfr-graphs'):
            os.makedirs(f'{prefix}temp/lfr-graphs')


        if method=='networkx':
            lfr_network = LFR_benchmark_graph(n[i], tau1[i], tau2[i], mu[i], average_degree=average_degree[i], min_community=min_community[i],seed=10)
            # lfr_network = LFR_benchmark_graph(n, tau1, tau2, mu[i],min_degree=min_degree,max_degree=max_degree , min_community=min_community,seed=10)
            communities = {frozenset(lfr_network.nodes[v]["community"]) for v in lfr_network}
            print(f"Number of communities: {len(communities)}")
            with open(f'{prefix}temp/lfr-graphs/lfr-communities-{i}.pkl',"wb") as f:
                pickle.dump([list(v) for v in communities],f)
            draw_LFR_graph(lfr_network,[list(v) for v in communities],f'{i}-LFR')
        elif method=='cdlib':
            # lfr_network,communities=generate_LFR(n[i], tau1[i], tau2[i], mu[i], average_degree[i], min_community[i])
            # lfr_network,communities=benchmark.LFR(n[i], tau1, tau2, mu,min_degree=min_degree,max_degree=max_degree , min_community=min_community,seed=10)
            lfr_network,communities=benchmark.LFR(n[i], tau1[i], tau2[i], mu[i], average_degree=average_degree[i], min_community=min_community[i],seed=10)
            
            print(f"Number of communities: {len(communities.communities)}")

            with open(f'{prefix}temp/lfr-graphs/lfr-communities-{i}.pkl',"wb") as f:
                pickle.dump(communities.communities,f)
            draw_LFR_graph(lfr_network,communities.communities,f'{i}-LFR')
            
        statistic_Q(f'{prefix}temp/statistics/q-lfr-non-overlapping.csv',i,lfr_network)
        

        print(f"Number of nodes: {lfr_network.number_of_nodes()}")
        print(f"Number of edges: {lfr_network.number_of_edges()}")

        
        with open(f'{prefix}temp/lfr-graphs/lfr-graph-{i}.pkl',"wb") as f:
            pickle.dump(lfr_network,f)
    return len(n)
        
        
        

def save_communities(community_obj, name,index):
    """
    使用pickle模块保存NodeClustering对象到文件。
    """
    if not os.path.exists(f'{prefix}temp/lfr-communities'):
        os.makedirs(f'{prefix}temp/lfr-communities')
    with open(f'{prefix}temp/lfr-communities/' +str(index)+"-"+ name + '.pkl', 'wb') as f:
        pickle.dump(community_obj, f)

def generate_communities(G,index):
    print(f'*** the {index}-th net is detected ***')

    # Girvan-Newman算法
    # print("Running Girvan-Newman...")
    # communities_gn = algorithms.girvan_newman(G, level=3)
    # draw_LFR_graph(G,communities_gn.communities,f'{index}-GirvanNewman')
    # print("Girvan-Newman is done.")
    # save_communities(communities_gn, 'Girvan-Newman',index)


    # Louvain算法
    print("Running Louvain...")
    communities_louvain = algorithms.louvain(G)
    draw_LFR_graph(G,communities_louvain.communities,f'{index}-louvain')
    print("Louvain is done.")
    save_communities(communities_louvain, 'Louvain',index)


    # CNM算法
    print("Running CNM...")
    communities_cnm = algorithms.greedy_modularity(G)
    draw_LFR_graph(G,communities_cnm.communities,f'{index}-CNM')
    print("CNM is done.")
    save_communities(communities_cnm, 'CNM',index)

    # LPA算法
    print("Running LPA...")
    communities_lpa = algorithms.label_propagation(G)
    draw_LFR_graph(G,communities_lpa.communities,f'{index}-LPA')
    print("LPA is done.")
    save_communities(communities_lpa, 'LPA',index)


    # Infomap算法
    print("Running Infomap...")
    communities_infomap = algorithms.infomap(G)
    draw_LFR_graph(G,communities_infomap.communities,f'{index}-Infomap')
    print("Infomap is done.")
    save_communities(communities_infomap, 'Infomap',index)


    # Eigenvector算法
    print("Running Eigenvector...")
    communities_eig = algorithms.eigenvector(G)
    draw_LFR_graph(G,communities_eig.communities,f'{index}-Eigenvector')
    print("Eigenvector is done.")
    save_communities(communities_eig, 'Eigenvector',index)


    # 计算平均社区数量
    avg_communities = round((+len(communities_louvain.communities)+len(communities_lpa.communities)+
                            len(communities_infomap.communities)+len(communities_eig.communities))/4)
    # print("avg_com:", avg_communities)

    # AGDL算法
    print("Running AGDL...")
    communities_agdl = algorithms.agdl(G, number_communities=avg_communities, kc=10)
    draw_LFR_graph(G,communities_agdl.communities,f'{index}-AGDL')
    print("AGDL is done.")
    save_communities(communities_agdl, 'AGDL',index)


    # EM算法
    print("Running EM...")
    communities_em = algorithms.em(G, k=avg_communities)
    draw_LFR_graph(G,communities_em.communities,f'{index}-EM')
    print("EM is done.")
    save_communities(communities_em, 'EM',index)

    print("All the communities are detected.")

def store_metrics_by_lfr_algo_groundtruth():
    output_csv_path=f"{prefix}outputs/graph-metrics-figcsv/"
    input_path=f"{prefix}outputs/lfr-ground-truth-metrics/"

    overlapping_alg = ["Kclique", "SLPA", "LEMON", "DEMON"]
    non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','AGDL','Eigenvector','EM','CNM']
    # non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','Eigenvector','EM','CNM']



    # 获取文件夹下所有文件和文件夹的名字
    filenames = os.listdir(input_path)
    filenames.sort()
    print(filenames)
    
        
    mean_list=[]
    
    for n_l in non_overlapping_alg:
        dict_list=[]
        
        for filename in filenames:
            with open(f'{input_path}{filename}', 'r', encoding='utf-8') as file:
                data2 = json.load(file)
            if bool(data2) and (f'{n_l}_groundtruth' in data2.keys()):
                dict_list.append(data2[f'{n_l}_groundtruth'])
        
        df_n = pd.DataFrame(dict_list)
        column_means = df_n.mean()
        column_means.name=n_l
        mean_list.append(column_means)
        df_n.to_csv(f'{output_csv_path}/lfr_{n_l}_metrics.csv')
    dff=pd.DataFrame({s.name: s for s in mean_list})
    dff.to_csv(f'{output_csv_path}lfr_non-overlapping-algo-metric.csv')


def lfr_metrics_with_ground_truth(G,index):

    if not os.path.exists(f'{prefix}outputs/lfr-ground-truth-metrics/'):
        os.mkdir(f'{prefix}outputs/lfr-ground-truth-metrics/')

    
    # alg = ['Girvan-Newman', 'Louvain', 'LPA', 'Infomap', 'AGDL', 'Eigenvector', 'EM','CNM']
    alg = [ 'Louvain', 'LPA', 'Infomap', 'CNM']

    
    dict = {}
    with open(f'{prefix}temp/lfr-graphs/lfr-communities-{index}.pkl', 'rb') as f:
        communities = pickle.load(f)
        ground_truth=NodeClustering(communities,G)

    for n_l1 in alg:
        # print(f'algo : **{n_l1}** and ground-truth : {index}')
        with open(f'{prefix}temp/lfr-communities/' +str(index)+"-"+ n_l1 + '.pkl', 'rb') as f:
            partition = pickle.load(f)
        if partition == "false":
            print(f'the community detection algo ** {n_l1} ** has failed!')
            continue
        # print(f'node number : {len(G.nodes())}')
        # print(f'partition one: {len([x for sublist in partition.communities for x in sublist])}')
        # print(f'ground-truth one : {len([x for sublist in ground_truth.communities for x in sublist])}')
        # print(f'partition node number : {len(set(itertools.chain.from_iterable(partition.communities)))}')
        # print(f'ground-truth node number : {len(set(itertools.chain.from_iterable(ground_truth.communities)))}')
        # print(partition.communities)
        # print(ground_truth.communities)

        partition_one=len([x for sublist in partition.communities for x in sublist])
        ground_truth_one=len([x for sublist in ground_truth.communities for x in sublist])
        if partition_one!=ground_truth_one:
            print(f'{n_l1} in {index} has different shape, partition shape is **{partition_one}** , ground_truth shape is **{ground_truth_one}**')
        else:
            dict[f'{n_l1}_groundtruth'] = main_non_overlapping_metrics_with_groundtruth(G, partition,ground_truth)


    # 将字典序列化为JSON格式的字符串
    json_data = json.dumps(dict)

    # 将JSON字符串写入到文件
    with open(f'{prefix}outputs/lfr-ground-truth-metrics/{index}-lfr-result.json', 'w') as f:
        f.write(json_data)

def main_lfr_network_generation(method):
    n=generate_lfr_benchmark(method)
    # n=4

    for i in range(0, n):
        with open(f'{prefix}temp/lfr-graphs/lfr-graph-{i}.pkl', 'rb') as f:
            G = pickle.load(f)
            # generate_communities(G,i)
            lfr_metrics_with_ground_truth(G,i)


main_lfr_network_generation('cdlib')

store_metrics_by_lfr_algo_groundtruth()

    
    
