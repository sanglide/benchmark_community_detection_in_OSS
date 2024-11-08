import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from gensim.models import Word2Vec
from node2vec import Node2Vec
import networkx as nx
from torch import Tensor 
# from stellargraph import StellarGraph
# from stellargraph.layer import GraphSAGE
# from stellargraph.mapper import GraphSAGELinkGenerator
# from tensorflow.keras import layers, models
from torch_geometric.nn import GAE, GCNConv
# from stellargraph import StellarGraph
# from stellargraph.layer import GraphSAGE
# from tensorflow.keras import layers, models
import random
import numpy as np
from sklearn.cluster import KMeans
from cdlib import NodeClustering
import os
import pickle
import argparse
import platform
system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("proj", help="the proj's name")
args = parser.parse_args()


def convert_G_to_data(G):
    # Step 2: Extract node features
    node_features = []
    for node in G.nodes():
        node_features.append([1 if i == node else 0 for i in G.nodes()])  # Get the feature for each node

    # Convert to a tensor
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Step 3: Extract edge index
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

    # Step 4: Create a Data object for PyTorch Geometric
    data = Data(x=node_features, edge_index=edge_index)

    return data,node_features,edge_index

class VGAE(torch.nn.Module):
    def __init__(self, num_features):
        super(VGAE, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def deepwalk(graph, num_walks=10, walk_length=5):
    walks = []
    nodes = list(graph.nodes())
    
    for node in nodes:
        for _ in range(num_walks):
            walk = [str(node)]  # Convert node to string for consistency
            for _ in range(walk_length - 1):
                current_node = walk[-1]
                neighbors = list(graph.neighbors(int(current_node)))  # Convert back to int for neighbors
                if neighbors:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    
    return walks

def save_communities(community_obj, name,proj):
    """
    使用pickle模块保存NodeClustering对象到文件。
    """
    if not os.path.exists(f'{prefix}temp/communities'):
        os.makedirs(f'{prefix}temp/communities')
    with open(f'{prefix}temp/communities/' +proj.replace("/","@")+"-"+ name + '.pkl', 'wb') as f:
        pickle.dump(community_obj, f)

def convert_nodeclustering_to_communities(node_clusters):
    communities=[]
    for i in range(max(node_clusters.values())+1):
        c=[]
        for node, cluster in node_clusters.items():
            if cluster==i:
                c.append(node)
        communities.append(c)
    return communities

def convert_tensor_to_communities(preds):
    co_mapping = {}
    count=0
    for i, community in enumerate(preds):
        if not int(community) in co_mapping.keys():
            co_mapping[int(community)]=count
            count+=1
    communities=[]
    for i in range(max(co_mapping.values())+1):
        c=[]
        for node, pred in enumerate(preds):
            if co_mapping[int(pred)]==i:
                c.append(node)
        communities.append(c)
    return communities

def main_deep_learning_method(G,proj):
    # todo:这个需要训练吗？数据到底是什么
    print(f" Running GCN ")
    # data = Data(x=node_features, edge_index=edge_index)

    data,node_features,edge_index=convert_G_to_data(G)
    model1 = GCN(num_features=node_features.shape[1], hidden_channels=16)
    # model1.load_state_dict(torch.load('gcn_model.pth'))
    # model1.eval()  # Set the model to evaluation mode
    embeddings1 = model1(data.x, data.edge_index)
    preds1 = embeddings1.argmax(dim=1)
    communities1=convert_tensor_to_communities(preds1)
    node_clustering1 = NodeClustering(communities1,graph=G)
    save_communities(node_clustering1, 'GCN',proj)

    print(f' Running Deep Walk ')
    # Generate random walks using DeepWalk
    walks = deepwalk(G, num_walks=10, walk_length=5)
    # Train Word2Vec model on the random walks
    model2 = Word2Vec(sentences=walks, vector_size=64, window=5, min_count=1, sg=1)
    # Get embeddings for each node
    embeddings = {str(node): model2.wv[str(node)] for node in G.nodes()}
    # Prepare the embeddings for clustering
    X = np.array(list(embeddings.values()))
    # Perform KMeans clustering
    num_clusters = 2  # Assuming we want to find 2 communities
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    # Assign clusters to nodes
    node_clusters2 = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}
    communities2=convert_nodeclustering_to_communities(node_clusters2)
    node_clustering2 = NodeClustering(communities2,graph=G)
    save_communities(node_clustering2, 'Deep-Walk',proj)
    
    
    print(f' Running Node2Vec ')
    # Generate embeddings
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model3 = node2vec.fit()
    # Get embeddings
    # Step 3: Get embeddings for each node
    embeddings = {node: model3.wv[node] for node in G.nodes()}

    # Step 4: Prepare the embeddings for clustering
    X = np.array(list(embeddings.values()))

    # Step 5: Perform KMeans clustering
    num_clusters3 = 2  # Specify the number of communities you want to find
    kmeans = KMeans(n_clusters=num_clusters3, random_state=0)
    kmeans.fit(X)

    # Assign clusters to nodes
    node_clusters3 = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}
    communities3=convert_nodeclustering_to_communities(node_clusters3)
    node_clustering3 = NodeClustering(communities3,graph=G)
    save_communities(node_clustering3, 'Node2vec',proj)



    # print(f' Running GraphSAGE ')
    # # Create a StellarGraph
    # G1 = StellarGraph.from_networkx(G)
    # # Create a GraphSAGE model
    # graphsage = GraphSAGE(layer_sizes=[32, 32], generator=generator, bias=True)
    # x_in = layers.Input(shape=(G1.node_features.shape[1],))
    # x_out = graphsage(x_in)
    # # Build and compile the model
    # model = models.Model(inputs=x_in, outputs=x_out)
    # model.compile(optimizer='adam', loss='binary_crossentropy')

    print(f' Running VGAE ')
    data,node_features,edge_index=convert_G_to_data(G)
    model4 = VGAE(num_features=node_features.shape[1])
    embeddings4 = model4.encode(data.x, data.edge_index)
    preds4 = embeddings4.argmax(dim=1)
    communities4=convert_tensor_to_communities(preds4)
    node_clustering4 = NodeClustering(communities4,graph=G)
    save_communities(node_clustering4, 'VGAE',proj)


#todo ：需要改掉
G= nx.read_graphml(f'{prefix}temp/graphs/{args.proj.replace("/","@")}-graph.graphml')
# Step 2: Create a mapping from string node names to integers
node_mapping = {node: i for i, node in enumerate(G.nodes())}

# Step 3: Create a new graph with integer node names
G_new = nx.Graph()

# Add edges to the new graph using the mapped integer node names
for u, v in G.edges():
    G_new.add_edge(node_mapping[u], node_mapping[v])
main_deep_learning_method(G_new,args.proj)