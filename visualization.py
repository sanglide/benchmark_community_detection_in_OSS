import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats, optimize
import statsmodels.api as sm



def draw_violin(data,color1,color2,label):    
    # Create the first violin plot for 'Avg'
    fig, ax1 = plt.subplots(figsize=(5, 6))
    sns.violinplot(data=data[['Avg']], ax=ax1, color=color1, inner='quartile')
    ax1.set_ylabel(f'Avg {label}', color='black',fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black',labelsize=20)

    # Create a second y-axis for 'Max'
    ax2 = ax1.twinx()
    sns.violinplot(data=data[['Max']], ax=ax2, color=color2, inner='quartile')
    ax2.set_ylabel(f'Max {label}', color='black',fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black',labelsize=20)


    # Set title and show the plot
    plt.tight_layout()
    plt.savefig(f"1-{label}.png")

def draw_violin2(data,color1,color2,color3,label):   
    # ['num cliques','max clique size','avg clique size'] 
    # Create the first violin plot for 'Avg'
    fig, ax1 = plt.subplots(figsize=(5, 6))
    sns.violinplot(data=data[['num cliques']], ax=ax1, color=color1, inner='quartile')
    ax1.set_ylabel(f'num cliques', color='black',fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black',labelsize=20)

    # Create a second y-axis for 'Max'
    ax2 = ax1.twinx()
    sns.violinplot(data=data[['max clique size','avg clique size']], ax=ax2, color=color2, inner='quartile')
    ax2.set_ylabel('max/avg clique size', color='black',fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black',labelsize=20)

    ax1.set_xticklabels(['num cliques', 'max clique size','avg clique size'], rotation=15)

    # Set title and show the plot
    plt.tight_layout()

    plt.savefig(f"1-{label}.png")

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_and_evaluate(x, y):
    # Linear fit
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    linear_r_squared = r_value**2
    
    # Exponential fit
    try:
        popt, _ = optimize.curve_fit(exp_func, x, y, p0=[1, 0.1, 1])
        exp_r_squared = 1 - (sum((y - exp_func(x, *popt))**2) / sum((y - np.mean(y))**2))
    except:
        popt = None
        exp_r_squared = 0
    
    # Determine best fit
    if linear_r_squared > 0.5 and linear_r_squared >= exp_r_squared:
        return 'linear', linear_r_squared, (slope, intercept)
    elif exp_r_squared > 0.5:
        return 'exp', exp_r_squared, popt
    else:
        return 'none', max(linear_r_squared, exp_r_squared), None

def scatter_chart(data):

    names=['closeness centrality','betweenness centrality','eigenvector centrality','average path length','num of nodes']
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(12, 12), squeeze=False)
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            ax = axes[i, j]
            
            if i != j:
                # Scatter plot
                ax.scatter(data[name2], data[name1], alpha=0.5)
            else:
                # Histogram on diagonal
                ax.hist(data[name1], bins=20, alpha=0.5)
            
            if i == n-1:
                ax.set_xlabel(name2)
            if j == 0:
                ax.set_ylabel(name1)
    
    plt.tight_layout()
    fig.suptitle('Scatter Plot Matrix of Network Weights with Trend Lines', y=1.02)
    plt.savefig(f"3-centrality-avgpath-numnodes.png")


def draw_node_degree_multifig(df):
    # Sample data creation
    # Replace this with your actual data
    num_networks = len(df)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    # 1. Bar plot for Average Node Degree
    sns.barplot(x=df.index, y='Average Node Degree', data=df, ax=axes[0], color='lightblue')
    axes[0].set_title(f'Avg Node Degree of {num_networks} Networks',fontdict={'fontsize': 10})
    axes[0].set_xlabel('Network Index')
    axes[0].set_ylabel('Average Node Degree')
    axes[0].set_xticklabels([])
    axes[0].set_xticks([])

    # 2. Scatter plot for Average Node Degree vs Power Law Exponent
    sns.scatterplot(x='Average Node Degree', y='Power Law Exponent', hue='Fits Power Law', data=df, ax=axes[1])
    axes[1].set_title('The correlation',fontdict={'fontsize': 10})
    axes[1].set_xlabel('Average Node Degree')
    axes[1].set_ylabel('Power Law Exponent')

    # 4. Count plot for networks fitting power law
    sns.countplot(x='Fits Power Law', data=df, ax=axes[2], palette='pastel')
    axes[2].set_title('Count of Networks Fitting Power Law',fontdict={'fontsize': 10},loc='right')
    axes[2].set_xlabel('Fits Power Law')
    axes[2].set_ylabel('Count')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'1-node-degree.png')

data_scatter=pd.read_csv("/outputs/graph-metrics-figcsv/centrality-avgpath-numnodes.csv")
data_violin_weight=pd.read_csv("/outputs/graph-metrics-figcsv/weight.csv")
data_violin_clique=pd.read_csv("/outputs/graph-metrics-figcsv/cliques.csv")

data_node_degree=pd.read_csv("/temp/statistics/communities-properties_graph.csv")
df_nodedegree =data_node_degree.sort_values(by='proj').drop_duplicates(subset='proj', keep='first')
df_nodedegree=df_nodedegree[["Average Node Degree","Power Law Exponent","KL Divergence","Fits Power Law"]]
draw_node_degree_multifig(df_nodedegree)


