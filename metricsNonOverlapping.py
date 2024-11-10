import networkx as nx
import numpy as np
from scipy.special import comb
from scipy.stats import entropy
from cdlib import evaluation
'''
<Metrics for Community Analysis: A Survey>
https://dl.acm.org/doi/10.1145/3091106
'''

'''
----------------------------------------------------
Simple Metrics Based on Topological Properties of the Network.
(1) Functions considering the internal connections only
(2) Functions considering the external connections only
(3) Functions considering internal and external connections
(4) Functions considering the model of a network
    Modularity: get_model_communities_modularity(G,partition)   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html

'''
def get_model_communities_modularity(G,partition):
    try:
        m=nx.community.modularity(G, partition)
    except:
        return ""
    return m
def convert_to_nx_partition(node_clustering):
    partition = {}
    for community_id, community in enumerate(node_clustering):
        partition[community_id]=community
    return partition

# Function to calculate the power mean of a community, M(w)
def power_mean(G,community, r):
    N_omega = len(community)
    mu_i = [G.degree(node) / N_omega for node in community]
    M_omega = (sum(mu ** r for mu in mu_i) / N_omega) ** (1 / r)
    return M_omega


# Function to calculate the score of a community, score(w)
def community_score(G,community, r):
    E_omega = G.subgraph(community).number_of_edges()
    M_omega = power_mean(G,community, r)
    score = M_omega * E_omega
    return score
def get_community_score(G,partition):
    # Group nodes by community
    communities = convert_to_nx_partition(partition)

    # Calculate Community Score with a chosen power mean order r
    r = 2  # You can choose a different value for r
    CS = sum(community_score(G,community, r) for community in communities.values())
    return CS


# Function to calculate SNode(v)
def strength_node(G, node, community):
    internal_degree = sum(1 for neighbor in G.neighbors(node) if neighbor in community)
    external_degree = sum(1 for neighbor in G.neighbors(node) if neighbor not in community)
    return (internal_degree - external_degree) / len(community)

# Function to calculate the strength of a community, SComm(w)
def strength_community(G, community):
    SComm = 0
    for v in community:
        SNode_v = strength_node(G, v, community)
        SComm += SNode_v
        for w in G.neighbors(v):
            if w in community:
                SNode_w = strength_node(G, w, community)
                SComm += 0.5 * SNode_w
    return SComm

# Function to calculate the overall fitness of the partition
def SPart(G, partition):
    communities = convert_to_nx_partition(partition)

    total_edges = G.number_of_edges()
    SPart_value = 0
    max_internal_edges={}
    for comm, nodes in communities.items():
        SComm_value = strength_community(G, nodes)
        if comm not in max_internal_edges:
            max_internal_edges={comm:len(nodes) * (len(nodes) - 1) / 2}
        v_omega = sum(1 for u, v in G.edges() if u in nodes and v in nodes) / total_edges
        SPart_value += SComm_value * v_omega / len(nodes)


    max_spart = sum(max_internal_edges[i] / G.number_of_edges() for i in max_internal_edges.keys())
    

    SPart=SPart_value / len(communities)

    normalized_spart = SPart / (max_spart*len(communities)) if max_spart > 0 else 0

    # print(f'max: {max_spart}, s part: {SPart} , normalise: {normalized_spart}')

    return SPart

def get_SPart(G,partition):

    # Calculate SPart
    spart_value = SPart(G, partition)



    return spart_value



# Function to calculate the density of a community
def community_density(G, community):
    subgraph = G.subgraph(community)
    n_omega = len(community)
    if n_omega <= 1:
        return 0
    E_omega = subgraph.number_of_edges()
    return 2 * E_omega / (n_omega * (n_omega - 1))

# Function to calculate the Kullback-Leibler divergence
# def kullback_leibler_divergence(q, p):
#     if p==0 or p==1:
#         return 0
#     else:
#         value=q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
#         return value

# Function to calculate the probability of a community, Pr(Ω)
def community_probability(G, community, p):
    n_omega = len(community)
    q = community_density(G, community)
    if q == 0:
        return 1
    D_q_p = entropy(p, q)
    return np.exp(-comb(n_omega, 2) * D_q_p)

# Function to calculate the significance of the graph
def significance(G, communities, p):
    log_prob = 0.0
    for community in communities.values():
        prob = community_probability(G, community, p)
        log_prob += np.log(prob)
    return -log_prob

def get_significance(G,partition):
    try:
        significance_value = evaluation.significance(G,partition)
    except:
        return 0

    return significance_value.score

def get_community(partition, node):
    for community in partition:
        if node in community:
            return community
    return None  # Return None if the node is not found in any community

# Function to calculate the internal neighbors of a vertex
def internal_neighbors(G, node, community):
    return sum(1 for neighbor in G.neighbors(node) if neighbor in community)

# Function to calculate the maximum number of neighbors to one of the external communities
def max_external_neighbors(G, node, partition):
    community = get_community(partition,node)
    external_communities = set(
        tuple(get_community(partition,neighbor)) for neighbor in G.neighbors(node) if get_community(partition,neighbor) != community)
    external_communities=[list(x) for x in external_communities]
    max_neighbors = 0
    for ext_comm in external_communities:
        ext_neighbors = sum(1 for neighbor in G.neighbors(node) if get_community(partition,neighbor) == ext_comm)
        if ext_neighbors > max_neighbors:
            max_neighbors = ext_neighbors
    return max_neighbors

# Function to calculate the internal clustering coefficient
def internal_clustering_coefficient(G, node, community):
    internal_neighbors = [neighbor for neighbor in G.neighbors(node) if neighbor in community]
    if len(internal_neighbors) < 2:
        return 0
    internal_subgraph = G.subgraph(internal_neighbors)
    possible_edges = len(internal_neighbors) * (len(internal_neighbors) - 1) / 2
    actual_edges = internal_subgraph.number_of_edges()
    return actual_edges / possible_edges


# Function to calculate the permanence of a vertex, Perm(v)
def permanence_vertex(G, node, partition):
    community = get_community(partition,node)
    I_v = internal_neighbors(G, node, community)
    E_max_v = max_external_neighbors(G, node, partition)
    d_v = G.degree(node)
    c_in_v = internal_clustering_coefficient(G, node, community)
    
    if E_max_v == 0:
        if d_v!=0:
            return I_v / d_v
        else:
            return I_v
    if d_v==0 and E_max_v!=0:
        return I_v/E_max_v-(1-c_in_v)
    return (I_v / E_max_v) * (1 / d_v) - (1 - c_in_v)

def get_permanence(G,partition):
    communities = convert_to_nx_partition(partition)

    # Calculate the permanence of the graph
    total_permanence = 0
    for node in G.nodes():
        total_permanence += permanence_vertex(G, node, partition)
    permanence_value = total_permanence / G.number_of_nodes()
    return permanence_value

# Function to calculate the maximum possible number of intra-community edges
def max_intra_community_edges(community):
    n = len(community)
    return n * (n - 1) // 2

# Function to calculate the Surprise metric
def surprise(G, communities):
    E = G.number_of_edges()
    F = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
    total_surprise = 0

    for community in communities.values():
        M = max_intra_community_edges(community)
        E_in = (G.subgraph(community)).number_of_edges()
        q = E_in / M if M > 0 else 0

        # Calculate the cumulative hypergeometric distribution
        surprise_value = 0
        for j in range(E_in, min(M, E) + 1):
            term = (comb(M, j) * comb(F - M, E - j)) / comb(F, E)
            surprise_value += term

        total_surprise += -np.log(surprise_value) if surprise_value > 0 else 0

    return total_surprise
def get_surprise(G,partition):
    communities = convert_to_nx_partition(partition)


    # Calculate the Surprise metric
    surprise_value = surprise(G, communities)
    return surprise_value

# Function to calculate the number of intra-community edges
def intra_community_edges(G, community):
    subgraph = G.subgraph(community)
    return subgraph.number_of_edges()

# Function to calculate the number of edges from the community to outside
def external_community_edges(G, community):
    external_edges = 0
    for node in community:
        for neighbor in G.neighbors(node):
            if neighbor not in community:
                external_edges += 1
    return external_edges

# Function to calculate the communitude of a community
def communitude(G, community):
    E_in = intra_community_edges(G, community)
    E_out = external_community_edges(G, community)
    m = G.number_of_edges()
    n_omega = len(community)
    if n_omega <= 1:
        return 0
    M = n_omega * (n_omega - 1) // 2
    if M == 0:
        return 0
    term1 = E_in / M
    term2 = (2 * (E_in + E_out) / (2 * m)) ** 2
    numerator = term1 - term2
    denominator = np.sqrt(term2 * (1 - term2))
    if denominator == 0:
        return 0
    return numerator / denominator

# Function to calculate the communitude of the graph
def communitude_graph(G, communities):
    total_communitude = 0
    for community in communities.values():
        total_communitude += communitude(G, community)
    return total_communitude / len(communities)
def get_communitude(G,partition):
    communities = convert_to_nx_partition(partition)


    communitude_value = communitude_graph(G, communities)
    return communitude_value

'''
----------------------------------------------------
3.1 Ground-Truth-Based Validation Metrics for Nonoverlapping Community Structure

'''


'''
----------------------------------------------------
'''
def main_non_overlapping_metrics(G,partition):
    dict={'N_Modularity':get_model_communities_modularity(G,partition.communities),
          'N_Community Score': get_community_score(G, partition.communities),
        'N_SPart': get_SPart(G, partition.communities),
        'N_Significance': get_significance(G, partition),
        'N_Permanence': get_permanence(G, partition.communities),
        'N_Surprise': get_surprise(G, partition.communities),
        'N_Communitude': get_communitude(G, partition.communities),
        }

    return dict