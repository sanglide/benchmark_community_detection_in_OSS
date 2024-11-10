# implementation of different evaluation metrics
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import networkx as nx
from cdlib import evaluation
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Process some project names.")
parser.add_argument('proj_name', type=str, default='1', help='The name of the project to process')
args = parser.parse_args()
proj = args.proj_name
if '/' in proj:
    proj = proj.split('/')[-1]


def load_communities(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def convert_communities_to_labels(communities):
    labels = {}
    for label, community in enumerate(communities):
        for node in community:
            # print(label, node)
            labels[node] = label
    labels = [labels[node] for node in sorted(labels)]
    return labels


def evaluate_nmi(community1, community2):
    nmi_score = evaluation.normalized_mutual_information(community1, community2)
    return nmi_score.score


def evaluate_ami(community1, community2):
    ami_score = evaluation.adjusted_mutual_information(community1, community2)
    return ami_score.score


def evaluate_ari(community1, community2):
    ari_score = evaluation.adjusted_rand_index(community1, community2)
    return ari_score.score


def evaluate_f1(community1, community2):
    f1_score = evaluation.nf1(community1, community2)
    return f1_score.score


def evaluate_weighted_modularity(graph, communities):
    modularity_score = evaluation.newman_girvan_modularity(graph, communities)
    return modularity_score.score


def apply_all_evaluation(communities):
    print("Evaluating the algorithms using Ground-Truth-Based Evaluation Metrics...")
    result_dic = {"NMI": {}, "AMI": {}, "ARI": {}, "F1": {}}
    for method in result_dic:
        for algo in communities:
            result_dic[method][algo] = {}

    for algo1 in communities:
        for algo2 in communities:
            if algo1 == algo2:
                nmi = ami = ari = f1 = 1
            else:
                community1 = communities[algo1]
                community2 = communities[algo2]
                nmi = evaluate_nmi(community1, community2)
                ami = evaluate_ami(community1, community2)
                ari = evaluate_ari(community1, community2)
                f1 = evaluate_f1(community1, community2)
            result_dic["NMI"][algo1][algo2] = nmi
            result_dic["AMI"][algo1][algo2] = ami
            result_dic["ARI"][algo1][algo2] = ari
            result_dic["F1"][algo1][algo2] = f1
    return result_dic


def apply_modularity_evaluation(graph, communities):
    print("Evaluating the algorithms using weighted modularity...")
    result_dic = {}
    for algo in communities:
        community = communities[algo]
        algo_modularity = evaluate_weighted_modularity(graph, community)
        result_dic[algo] = algo_modularity
    return result_dic


def draw_heatmap(method, evaluation_results):
    result = evaluation_results[method]
    algorithms = list(result.keys())
    similarity_matrix = np.zeros((len(algorithms), len(algorithms)))
    for i, algorithm1 in enumerate(algorithms):
        for j, algorithm2 in enumerate(algorithms):
            similarity_matrix[i, j] = result[algorithm1][algorithm2]

    # 绘制热图
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Similarity Score')
    plt.xticks(np.arange(len(algorithms)), algorithms, rotation=45)
    plt.yticks(np.arange(len(algorithms)), algorithms)
    plt.title('Similarity between Community Detection Algorithms')
    plt.tight_layout()
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("outputs/pics"):
        os.makedirs("outputs/pics")
    save_path = "outputs/pics/" + method + ".jpg"
    plt.savefig(save_path, dpi=300)
    print("Saved the heatmap of the evaluation metric ", method)


def draw_bar_graph(method, modularity_result):
    algorithms = list(modularity_result.keys())
    scores = list(modularity_result.values())

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, scores, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Algorithm')
    plt.ylabel('Weighted Modularity Score')
    plt.title('Comparison of Community Detection Algorithms by Weighted Modularity')
    plt.ylim(0, max(scores) + 0.1)
    plt.xticks(rotation=45)

    # 显示每个柱的具体分数
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    save_path = f"outputs/pics/{method}.jpg"
    plt.savefig(save_path, dpi=300)
    print("Saved the bar graph of the weighted modularity evaluation")


G = nx.read_graphml('temp/graphs/graph.graphml')
all_communities = {
    "Girvan-Newman": load_communities('temp/communities/Girvan-Newman.pkl'),
    # "CNM": load_communities('temp/communities/CNM.pkl'),
    "Louvain": load_communities('temp/communities/Louvain.pkl'),
    "LPA": load_communities('temp/communities/LPA.pkl'),
    "Infomap": load_communities('temp/communities/Infomap.pkl'),
    "AGDL": load_communities('temp/communities/AGDL.pkl'),
    "Eigenvector": load_communities('temp/communities/Eigenvector.pkl'),
    "EM": load_communities('temp/communities/EM.pkl')
}
evaluation_result = apply_all_evaluation(all_communities)
modularity_result = apply_modularity_evaluation(G, all_communities)
# print(evaluation_result)
# print(modularity_result)


for method in evaluation_result:
    draw_heatmap(method, evaluation_result)
draw_bar_graph("Modularity", modularity_result)

evaluation_result["Modularity"] = modularity_result
with open(f'outputs/non_overlap_res_{proj}.json', 'w') as json_file:
    json.dump(evaluation_result, json_file, indent=4)


########################################################################


def evaluate_overlapping_modularity(graph, communities):
    modularity_score = evaluation.modularity_overlap(graph, communities)
    return modularity_score.score


def evaluate_omega_index(community1, community2):
    omega_score = evaluation.omega(community1, community2)
    return omega_score.score


def evaluate_onmi(community1, community2):
    onmi_score = evaluation.overlapping_normalized_mutual_information_LFK(community1, community2)
    return onmi_score.score


def apply_all_overlapping_evaluation(communities):
    print("Evaluating overlapping community detection algorithms...")
    result_dic = {"Overlapping Modularity": {}, "OMEGA Index": {}, "ONMI": {}}
    for method in result_dic:
        for algo in communities:
            result_dic[method][algo] = {}

    G = nx.read_graphml('temp/graphs/graph.graphml')
    # G = nx.karate_club_graph()

    for algo1 in communities:
        community1 = communities[algo1]
        result_dic["Overlapping Modularity"][algo1] = evaluate_overlapping_modularity(G, community1)

        for algo2 in communities:
            community2 = communities[algo2]
            if algo1 == algo2:
                omega = onmi = 1
            else:
                omega = evaluate_omega_index(community1, community2)
                onmi = evaluate_onmi(community1, community2)
            result_dic["OMEGA Index"][algo1][algo2] = omega
            result_dic["ONMI"][algo1][algo2] = onmi

    return result_dic
def ensure_full_coverage(community_obj):
    all_nodes = community_obj.graph.nodes
    covered_nodes = set()
    for community in community_obj.communities:
        covered_nodes.update(community)

    uncovered_nodes = all_nodes - covered_nodes
    new_communities = community_obj.communities.copy()
    if uncovered_nodes:
        new_communities.append(list(uncovered_nodes))

    from cdlib import NodeClustering
    return NodeClustering(new_communities, graph=community_obj.graph, method_name=community_obj.method_name,
                          method_parameters=community_obj.method_parameters)


all_communities = {
    "Kclique": load_communities('temp/communities/Kclique.pkl'),
    "SLPA": load_communities('temp/communities/SLPA.pkl'),
    # "CONGA": load_communities('temp/communities/CONGA.pkl'),
    "LEMON": load_communities('temp/communities/LEMON.pkl'),
    "DEMON": load_communities('temp/communities/DEMON.pkl'),
}

all_communities = {name: ensure_full_coverage(comm) for name, comm in all_communities.items()}

evaluation_result = apply_all_overlapping_evaluation(all_communities)

for method in ["OMEGA Index", "ONMI"]:
    draw_heatmap(method, evaluation_result)
draw_bar_graph('Modularity overlap', evaluation_result["Overlapping Modularity"])

evaluation_result["modularity"] = modularity_result
with open(f'outputs/overlap_res_{proj}.json', 'w') as json_file:
    json.dump(evaluation_result, json_file, indent=4)