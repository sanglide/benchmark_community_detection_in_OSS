from cdlib import algorithms
import pickle
import platform
import community as community_louvain
import os
import pandas as pd
import networkx as nx
from metricsNonOverlapping import main_non_overlapping_metrics
import json
from cdlib import NodeClustering
from networkx.algorithms.community import is_partition
import numpy as np
from sklearn import preprocessing


system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

def convert_cdlib(partition):
    user_list=list(partition.keys())
    community_list=set(partition.values())
    print(partition)
    # print(community_list)
    communities=[]
    for c in community_list:
        cc_temp=[]
        for u in user_list:
            if partition[u]==c:
                cc_temp.append(user_list.index(u))
        communities.append(cc_temp)
    return communities

def get_commuties_by_resolution(G,resolution,proj_name,method):

    try:
        if method=='Louvain':
            partition=nx.community.louvain_communities(G, resolution=resolution)
        else:
            partition=nx.community.greedy_modularity_communities(G, resolution=resolution)
    except:
        return "not partition"

    if (not is_partition(G, partition)) or len(partition)==0 :
        return "not partition"

    community=[list(c) for c in partition]
    communities=NodeClustering(community,G)
    dict=main_non_overlapping_metrics(G,communities)
    
    return dict

def main_resolution_real_world_exp():

    resolutions=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # resolutions=[1]

    input_path=f'{prefix}temp/graphs'
    filenames = os.listdir(input_path)
    filenames=[filename for filename in filenames if 'directed' not in filename]
    min_max_scaler = preprocessing.MinMaxScaler()
    name_list=['N_Modularity','N_Community Score','N_SPart','N_Significance','N_Permanence','N_Surprise','N_Communitude']
    print(f'there is {len(filenames)} proj will be detected')

    if not os.path.exists(f'{prefix}outputs/resolution-exp/'):
        os.mkdir(f'{prefix}outputs/resolution-exp/')

    dict_list=[]
    count=0
    for filename in filenames:
        G = nx.read_graphml(f'{input_path}/{filename}')
        dict=get_commuties_by_resolution(G,1,filename[:len(filename)-14],'Louvain')
        if dict=='not partition':
            count+=1
        else:
            dict_list.append(dict)
    df_ground_truth=pd.DataFrame(dict_list)
    for name in name_list:
        df_ground_truth[name]=min_max_scaler.fit_transform(df_ground_truth[name].values.reshape(-1,1))
    
    mean_list=[]
    std_list=[]
    rmse_list=[]
    method='CNM'
    for resolution in resolutions:
        count=0
        dict_list=[]
        for filename in filenames:
            G = nx.read_graphml(f'{input_path}/{filename}')
            dict=get_commuties_by_resolution(G,resolution,filename[:len(filename)-14],method)
            if dict=='not partition':
                count+=1
            else:
                dict_list.append(dict)
        print(f'resolution {resolution} has {count} project cannot get partition by {method}')
        df_one_resolution=pd.DataFrame(dict_list)
        
        for name in name_list:
            df_one_resolution[name]=min_max_scaler.fit_transform(df_one_resolution[name].values.reshape(-1,1))
        df_one_resolution.to_csv(f'{prefix}outputs/resolution-exp/{resolution}_all_file.csv')

        rmse_modularity = np.sqrt(((df_ground_truth['N_Modularity'] - df_one_resolution['N_Modularity']) ** 2).mean())
        rmse_cs=np.sqrt(((df_ground_truth['N_Community Score'] - df_one_resolution['N_Community Score']) ** 2).mean())
        rmse_sp=np.sqrt(((df_ground_truth['N_SPart'] - df_one_resolution['N_SPart']) ** 2).mean())
        rmse_significance=np.sqrt(((df_ground_truth['N_Significance'] - df_one_resolution['N_Significance']) ** 2).mean())
        rmse_permanence=np.sqrt(((df_ground_truth['N_Permanence'] - df_one_resolution['N_Permanence']) ** 2).mean())
        rmse_surprise=np.sqrt(((df_ground_truth['N_Surprise'] - df_one_resolution['N_Surprise']) ** 2).mean())
        rmse_communitude=np.sqrt(((df_ground_truth['N_Communitude'] - df_one_resolution['N_Communitude']) ** 2).mean())

        column_means = df_one_resolution.mean()
        column_stds=df_one_resolution.std()
        column_rmse=pd.Series({'N_Modularity':rmse_modularity,'N_Community Score':rmse_cs,'N_SPart':rmse_sp,
                               'N_Significance':rmse_significance,'N_Permanence':rmse_permanence,'N_Surprise':rmse_surprise,'N_Communitude':rmse_communitude,})

        column_stds.name=resolution
        column_means.name=resolution
        column_rmse.name=resolution

        mean_list.append(column_means)
        std_list.append(column_stds)
        rmse_list.append(column_rmse)
    dff=pd.DataFrame({s.name: s for s in mean_list})
    dff=dff.transpose()

    dfff=pd.DataFrame({s.name: s for s in std_list})
    dfff=dfff.transpose()

    dffff=pd.DataFrame({s.name: s for s in rmse_list})
    dffff=dffff.transpose()

    # print(dfff)
    # print(dffff)

    df_latex = dff.applymap('{:.2f}'.format).astype(str) + '(' + dfff.applymap('{:.2f}'.format).astype(str) + ')'+ '(' + dffff.applymap('{:.2f}'.format).astype(str) + ')'
    df_latex.to_csv(f'{prefix}outputs/resolution-exp/all_file_all_resolution.csv')
    dffff.to_csv(f'{prefix}outputs/resolution-exp/rmse.csv')

def main_resolution_LFR_exp():
    # LFR benchmark
    pass


main_resolution_real_world_exp()