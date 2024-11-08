'''
1. filter the user whose name contains 'bot'
2. cut the data by 'proj' and save
'''
import argparse
import platform
import pandas as pd

system = platform.system()
if system == 'Windows':
    prefix=""
elif system == 'Linux':
    prefix="benchmark_community_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("issue_pr_df_path", help="source data path")
args = parser.parse_args()



def convert_data(path):
    # Read the CSV file
    df = pd.read_csv(path)

    # Print the initial row count
    print(f"Initial row count: {len(df)}")
    print(f"the data is from {df['startTime'].min()} to {df['startTime'].max()}")
    print(f"there is {df['proj'].nunique()} repo")
    proj_before=pd.DataFrame(df['proj'].drop_duplicates(keep='last'))

    # Filter out rows containing 'bot' in 'createUser' or 'commentsUser' columns
    df_filter = df[~df['createUser'].str.contains('bot', case=False, na=False) &
            ~df['commentsUser'].str.contains('bot', case=False, na=False)]

    # Print the row count after filtering
    print(f"Row count after remove **bots** filtering: {len(df_filter)}")
    print(f"there is {df_filter['proj'].nunique()} repo after remove bots")

    # Group by 'proj' and count the number of rows for each project
    project_counts = df_filter.groupby('proj').size()
    projects_with_more_than_3_rows = project_counts[project_counts > 3].index
    df_filter = df_filter[df_filter['proj'].isin(projects_with_more_than_3_rows)]

    print(f"there is {df_filter['proj'].nunique()} repo after data filtering")

    proj_after=pd.DataFrame(df_filter['proj'].drop_duplicates(keep='last'))

    filter_proj=proj_before[~proj_before['proj'].isin(proj_after['proj'])]
    print(f'delete {len(filter_proj)} projects')

    # df_delete=df[df['proj'].isin(filter_proj['proj'])]
    # for proj, group in df_delete.groupby('proj'):
    #     print(f'repo name : {proj}')
    #     print(group)


    return df_filter

def group_and_save_by_proj(df):

    # Group the dataframe by 'proj' and save as separate CSV files
    projects=[]
    count=0
    for proj, group in df.groupby('proj'):
        filename = f"{prefix}data/{proj.replace('/','@')}.csv"
        group.to_csv(filename, index=False)
        projects.append(proj)
        count=count+1
    proj_df=pd.DataFrame(projects)
    proj_df.columns=["proj"]
    proj_df.to_csv(f"{prefix}temp/project_list_filter.csv",index=False)
    print(f"Saved {count} proj")

# ... (rest of the code remains the same)


# Read file from the same folder
df = convert_data(args.issue_pr_df_path)
group_and_save_by_proj(df)  # Add this line to group and save the data

