import networkx as nx
import numpy as np
'''
<Metrics for Community Analysis: A Survey>
https://dl.acm.org/doi/10.1145/3091106
'''

'''
----------------------------------------------------
2.2 Metrics for Overlapping and Fuzzy Community Detection
(1) Modularity : compute_modularity(G, F), using Equation (51) in the paper above
(2) Flex: compute_flex(G, partition, lambda_param=0.5, kappa=0.1, gamma=0.01)
'''
def compute_modularity(G, partition):
    """
    Compute the modularity of overlapping communities in graph G.

    Parameters:
    - G: A networkx graph.
    - F: A dictionary where keys are nodes and values are dictionaries.
         Each sub-dictionary maps community identifiers to membership strengths of the node in those communities.

    Returns:
    - Q_ov: The modularity value.
    """

    F = {}
    for community_id, community in enumerate(partition):
        for node in community:
            if node not in F:
                F[node] = {}
            F[node][community_id] = 1.0  # Assuming binary membership
    # print(f'F : {F}')

    A = nx.adjacency_matrix(G).todense()
    
    nodes_G=set(G.nodes())
    nodes_actual = list(set(item for sublist in partition for item in sublist))

    # nodes_to_remove = [node for node in G.nodes() if node not in nodes]
    # G.remove_nodes_from(nodes_to_remove)

    m = G.number_of_edges()
    n_G = G.number_of_nodes()
    n_actual=len(nodes_actual)

    # Degree of each node
    # print(f'{[G.degree(node) for node in nodes]}')
    k = np.array([G.degree(node) for node in nodes_G])

    # Number of communities each node belongs to
    O = np.array([len(F[node]) for node in nodes_actual])

    # Initialize modularity
    Q_ov = 0

    # Compute the modularity
    for i in range(n_actual):
        for j in range(n_actual):
            if i != j:
                # Sum over all communities
                sum_Fic_Fjc = sum(
                    F[nodes_actual[i]].get(c, 0) * F[nodes_actual[j]].get(c, 0) for c in set(F[nodes_actual[i]]).intersection(F[nodes_actual[j]]))
                Q_ov += (A[i, j] - (k[i] * k[j]) / (2 * m)) * (sum_Fic_Fjc / (O[i] * O[j]))

    Q_ov /= (2 * m)
    return Q_ov


def triangle_ratio(G,node, community):
    triangles = sum(1 for neighbor in G[node] if neighbor in community and any(
        other_neighbor in community for other_neighbor in G[neighbor] if other_neighbor != node))
    total_triangles = sum(1 for neighbor in G[node] for other_neighbor in G[neighbor] if other_neighbor != node) / 2
    return triangles / total_triangles if total_triangles > 0 else 0

def neighbor_fraction(G,node, community):
    internal_neighbors = sum(1 for neighbor in G.neighbors(node) if neighbor in community)
    total_neighbors = G.degree(node)
    return internal_neighbors / total_neighbors if total_neighbors > 0 else 0


def wedge_ratio(G,node, community):
    wedges = sum(1 for neighbor in G.neighbors(node) for second_neighbor in G.neighbors(neighbor) if
                 second_neighbor != node and second_neighbor in community)
    total_wedges = sum(
        1 for neighbor in G.neighbors(node) for second_neighbor in G.neighbors(neighbor) if second_neighbor != node)
    return wedges / total_wedges if total_wedges > 0 else 0

def compute_flex(G, partition, lambda_param=0.5, kappa=0.1, gamma=0.01):
    """
    Compute the Flex metric for overlapping communities in a graph G.

    Parameters:
    - G: A networkx graph.
    - partition: A dictionary where keys are community identifiers and values are lists of nodes in those communities.
    - lambda_param: Weight for the triangle ratio term.
    - kappa: Weight for the wedge ratio term.
    - gamma: Penalization weight to avoid trivial solutions.

    Returns:
    - Flex value for the community structure.
    """
    N = len(G.nodes())

    # Compute Community Contribution for each community
    CC_values = []
    for community_id, nodes in  enumerate(partition):
        CC = 0
        for node in nodes:
            delta = triangle_ratio(G,node, nodes)
            N_i = neighbor_fraction(G,node, nodes)
            Lambda = wedge_ratio(G,node, nodes)
            LC = lambda_param * delta + (1 - lambda_param) * N_i - kappa * Lambda
            CC += LC
        CC -= gamma * len(nodes) / N
        CC_values.append(CC)

    # Compute Flex
    Flex = sum(CC_values) / N
    return Flex


'''
----------------------------------------------------
'''
def main_overlapping_metrics(G,partition):
    dict={
        # "Modularity":compute_modularity(G,partition.communities),
        "Modularity":nx.community.modularity(G, partition.communities),
        "Flex":compute_flex(G, partition.communities)
    }
    return dict