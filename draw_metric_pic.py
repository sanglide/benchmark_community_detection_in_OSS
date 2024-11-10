import os
import platform
import json
import pandas as pd
import csv
import itertools
from sklearn import preprocessing

system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

def get_file_list(folder_path,key):
    list=[]
    for filename in os.listdir(folder_path):
        with open(f'{folder_path}{filename}', 'r', encoding='utf-8') as file:
            data = json.load(file)
            list.append(data[key])
    return list
def get_CS_power_law():
    input_path=f'{prefix}temp/statistics/communities-properties.csv'
    output_path=f"{prefix}outputs/graph-metrics-figcsv/CS_ispower_table.csv"

    df=pd.read_csv(input_path)
    grouped = df.groupby('algo')

    # Count the number of each group
    group_counts = grouped.size().reset_index(name='total_count')

    # Count the 'TRUE' values in 'ND_ispowerlaw' column for each group
    true_counts = grouped['CS_is_powerlaw'].apply(lambda x: (x == True).sum()).reset_index(name='true_count')

    # Merge the two results
    result = pd.merge(group_counts, true_counts, on='algo')

    # Calculate the percentage of TRUE values
    result['true_percentage'] = (result['true_count'] / result['total_count'] * 100).round(2)

    # Sort the result by the total count in descending order
    result = result.sort_values('total_count', ascending=False)

    print(result)
    result.to_csv(output_path)

def draw_node_degree_of_algo(algo1):
    input_path=f'{prefix}temp/statistics/communities-properties.csv'
    output_path=f"{prefix}outputs/graph-metrics-figcsv/ND_comparison_table.csv"

    df=pd.read_csv(input_path)
    df_filter1=df[df['algo']==algo1]

    df_filter1=df_filter1[['average degree','ND_alpha','ND_ks','proj']]

    df_filter1.to_csv(output_path)

def histogram_pic():
    output_csv_path=f"{prefix}outputs/graph-metrics-figcsv/"
    input_path=f"{prefix}outputs/graph-metrics/"

    if not os.path.exists(output_csv_path):
        os.mkdir(output_csv_path)
    
    list_numnodes=get_file_list(input_path,'num_nodes')
    # df_numnodes = pd.DataFrame(list_numnodes)
    # df_numnodes.to_csv(f'{output_csv_path}numOfNodes.csv', index=False)

    list_cc=get_file_list(input_path,'avg_clustering_coefficients')
    df_cc = pd.DataFrame(list_cc)
    df_cc.to_csv(f'{output_csv_path}avg_clustering_coefficients.csv', index=False)


    list_density=get_file_list(input_path, "density")
    df_density = pd.DataFrame(list_density)
    df_density.to_csv(f'{output_csv_path}density.csv', index=False)

    list_diameter=get_file_list(input_path, "diameter")
    df_diameter = pd.DataFrame(list_diameter)
    df_diameter.to_csv(f'{output_csv_path}diameter.csv', index=False)

    list_average_path_length=get_file_list(input_path, "average_path_length")
    # df_average_path_length = pd.DataFrame(list_average_path_length)
    # df_average_path_length.to_csv(f'{output_csv_path}average_path_length.csv', index=False)


    list_closeness_centrality=get_file_list(input_path,'avg_closeness_centrality')
    list_betweenness_centrality=get_file_list(input_path,'avg_betweenness_centrality')
    list_eigenvector_centrality=get_file_list(input_path,'avg_eigenvector_centrality')
                                              
    rows = zip(list_closeness_centrality, list_betweenness_centrality, list_eigenvector_centrality,list_average_path_length,list_numnodes)

    # Write to CSV
    with open(f'{output_csv_path}centrality-avgpath-numnodes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    

    list_cliques_num=get_file_list(input_path,'num_cliques')
    list_min_cliques_size=get_file_list(input_path,'min_clique_size')
    list_max_cliques_size=get_file_list(input_path,'max_clique_size')
    list_avg_cliques_size=get_file_list(input_path,'avg_clique_size')
                                              
    rowss = zip(list_cliques_num,list_min_cliques_size,list_max_cliques_size,list_avg_cliques_size)

    # Write to CSV
    with open(f'{output_csv_path}cliques.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rowss)
    
    
    list_min_weight=get_file_list(input_path,'min_weight')
    list_max_weight=get_file_list(input_path,'max_weight')
    list_avg_weight=get_file_list(input_path,'avg_weight')
                                              
    rowsss = zip(list_min_weight,list_max_weight,list_avg_weight)

    # Write to CSV
    with open(f'{output_csv_path}weight.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rowsss)
    
    list_r=get_file_list(input_path,"network_robustness")
    list_robustness_1=[x["node_connectivity"] for x in list_r]
    list_robustness_2=[x["edge_connectivity"] for x in list_r]
    list_robustness_3=[x["robustness_score"] for x in list_r]
                                              
    rowssss = zip(list_robustness_1,list_robustness_2,list_robustness_3)

    # Write to CSV
    with open(f'{output_csv_path}robustness.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rowssss)


def store_metrics_by_algo():
    output_csv_path=f"{prefix}outputs/graph-metrics-figcsv/"
    input_path=f"{prefix}outputs/metrics/"

    overlapping_alg=["Kclique","SLPA","LEMON","DEMON",'CONGA']
    non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','AGDL','Eigenvector','EM','CNM']


    filenames = os.listdir(input_path)

    dict_list=[]        
    for filename in filenames:
        with open(f'{input_path}{filename}', 'r', encoding='utf-8') as file:
            data1 = json.load(file)
        for o_l in overlapping_alg:
            if bool(data1) and (o_l in data1.keys()):
                temp=data1[o_l]
                temp['algo']=o_l
                dict_list.append(temp)
    df_o = pd.DataFrame(dict_list)
    min_max_scaler = preprocessing.MinMaxScaler()
    df_o['Modularity_scale'] = min_max_scaler.fit_transform(df_o['Modularity'].values.reshape(-1,1))
    df_o['Flex_scale'] = min_max_scaler.fit_transform(df_o['Flex'].values.reshape(-1,1))

    df_o.to_csv(f'{output_csv_path}overlapping-algo-metric.csv')
    
    dict_list=[]    
    for filename in filenames:
        with open(f'{input_path}{filename}', 'r', encoding='utf-8') as file:
            data2 = json.load(file)
        for n_l in non_overlapping_alg:
            if bool(data2) and (n_l in data2.keys()):
                temp=data2[n_l]
                temp['algo']=n_l
                dict_list.append(temp) 
    df_n = pd.DataFrame(dict_list)
    min_max_scaler = preprocessing.MinMaxScaler()
    df_n['N_SPart_scale'] = min_max_scaler.fit_transform(df_n['N_SPart'].values.reshape(-1,1))
    df_n['Communitude_scale']=min_max_scaler.fit_transform(df_n['N_Communitude'].values.reshape(-1,1))
    df_n['Modularity_scale']=min_max_scaler.fit_transform(df_n['N_Modularity'].values.reshape(-1,1))
    df_n['Community Score_scale']=min_max_scaler.fit_transform(df_n['N_Community Score'].values.reshape(-1,1))
    df_n['Significance_scale']=min_max_scaler.fit_transform(df_n['N_Significance'].values.reshape(-1,1))
    df_n['Permanence_scale']=min_max_scaler.fit_transform(df_n['N_Permanence'].values.reshape(-1,1))
    df_n['Surprise_scale']=min_max_scaler.fit_transform(df_n['N_Surprise'].values.reshape(-1,1))
    df_n.to_csv(f'{output_csv_path}non-overlapping-algo-metric.csv')


def store_metrics_by_algo_groundtruth():

    output_csv_path=f"{prefix}outputs/graph-metrics-figcsv/"
    input_path=f"{prefix}outputs/ground-truth-metrics/"
    overlapping_alg=["Kclique","SLPA","LEMON","DEMON"]
    non_overlapping_alg=['Girvan-Newman','Louvain','LPA','Infomap','AGDL','Eigenvector','EM','CNM']
    filenames = os.listdir(input_path)

    mean_list_o=[]
    std_list_o=[]
    for o_l1,o_l2 in list(itertools.product(overlapping_alg, repeat=2)):
        dict_list=[]
        for filename in filenames:
            with open(f'{input_path}{filename}', 'r', encoding='utf-8') as file:
                data1 = json.load(file)
            if bool(data1) and (f'{o_l1}_{o_l2}' in data1.keys()):
                dict_list.append(data1[f'{o_l1}_{o_l2}'])
        df_o = pd.DataFrame(dict_list)

        column_means = df_o.mean()
        column_stds=df_o.std()
        column_stds.name=f'{o_l1}_{o_l2}'
        column_means.name=f'{o_l1}_{o_l2}'
        mean_list_o.append(column_means)
        std_list_o.append(column_stds)
    dff=pd.DataFrame({s.name: s for s in mean_list_o})
    dff=dff.transpose()
    dff.to_csv(f'{output_csv_path}ground-truth-overlapping-algo-metric.csv')

    dfff=pd.DataFrame({s.name: s for s in std_list_o})
    dfff=dfff.transpose()
    dfff.to_csv(f'{output_csv_path}ground-truth-overlapping-algo-metric-std.csv')

    df_latex = dff.applymap('{:.2f}'.format).astype(str) + '(' + dfff.applymap('{:.2f}'.format).astype(str) + ')'
    df_latex.to_csv(f'{output_csv_path}ground-truth-overlapping-algo-metric-latex.csv')



    mean_list_n=[]
    std_list_n=[]
    for n_l1,n_l2 in list(itertools.product(non_overlapping_alg, repeat=2)):
        dict_list=[]
        for filename in filenames:
            with open(f'{input_path}{filename}', 'r', encoding='utf-8') as file:
                data1 = json.load(file)
            if bool(data1) and (f'{n_l1}_{n_l2}' in data1.keys()):
                dict_list.append(data1[f'{n_l1}_{n_l2}'])
        df_n = pd.DataFrame(dict_list)

        column_means = df_n.mean()
        column_stds=df_n.std()
        column_stds.name=f'{n_l1}_{n_l2}'
        column_means.name=f'{n_l1}_{n_l2}'
        mean_list_n.append(column_means)
        std_list_n.append(column_stds)
    dff=pd.DataFrame({s.name: s for s in mean_list_n})
    dff=dff.transpose()
    dff.to_csv(f'{output_csv_path}ground-truth-non-overlapping-algo-metric-mean.csv')

    dfff=pd.DataFrame({s.name: s for s in std_list_n})
    dfff=dfff.transpose()
    dfff.to_csv(f'{output_csv_path}ground-truth-non-overlapping-algo-metric-std.csv')

    df_latex = dff.applymap('{:.2f}'.format).astype(str) + '(' + dfff.applymap('{:.2f}'.format).astype(str) + ')'
    df_latex.to_csv(f'{output_csv_path}ground-truth-non-overlapping-algo-metric-latex.csv')



histogram_pic()
# get_CS_power_law()
# draw_node_degree_of_algo('LPA','Infomap')

store_metrics_by_algo()
# store_metrics_by_algo_groundtruth()