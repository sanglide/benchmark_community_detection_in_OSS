import os
import subprocess
import pandas as pd
import logging
import platform
from collections import defaultdict
import json



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_script(script_path, *args):
    """
    运行位于 script_path 的 Python 脚本，附带 args 参数。
    """
    command = ["python", script_path] + list(args)
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully running {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}")

def count_files_in_directory(directory):
    count=len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    return f'There are {count} files in the {directory}.\n'


def statistic_projects_methods(folder_path):
    method_counts = defaultdict(set)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            parts = filename.split('-')
            if len(parts) >= 2:
                project_name = parts[0]
                method_name = '-'.join(parts[1:]).rsplit('.', 1)[0]
                method_counts[method_name].add(project_name)
    dict_str=""
    for key, value in method_counts.items():
        dict_str+=f"method: {key}, proj count: {len(value)}\n"
    return dict_str

def statastic_project_running(prefix):
    s=""
    s+=count_files_in_directory(prefix+"data/")
    s+=count_files_in_directory(prefix+"temp/communities")
    s+=count_files_in_directory(prefix+"temp/graphs")
    s+=count_files_in_directory(prefix+"outputs/metrics")
    s+=statistic_projects_methods(prefix+"temp/communities")

    logging.basicConfig(filename=prefix+'app.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(s)

def delete_file(file_path):
    try:
        # Check if the file exists
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file
            print(f"File '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")

if __name__ == '__main__':
    # 加载项目列表和数据文件
    system = platform.system()
    if system == 'Windows':
        project_df = pd.read_csv('../data-2964/project_list.csv')
        issue_pr_df_path = '../data-2964/issue_pr_data.csv'
        prefix=""
    elif system == 'Linux':
        project_df = pd.read_csv('./data-2964/project_list.csv')
        issue_pr_df_path = './data-2964/issue_pr_data.csv'
        prefix="benchmark_community_detection/"
    else:
        project_df = pd.read_csv('')
        issue_pr_df_path = ''
        prefix=""

    if not os.path.exists(f'{prefix}outputs/'):
        os.mkdir(f'{prefix}outputs/')
    if not os.path.exists(f'{prefix}data/'):
        os.mkdir(f'{prefix}data/')
    if not os.path.exists(f'{prefix}temp/'):
        os.mkdir(f'{prefix}temp/')
    delete_file(f'{prefix}temp/statistics/communities-properties.csv')
    delete_file(f'{prefix}temp/statistics/q.csv')
    delete_file(f'{prefix}temp/statistics/communities-properties_graph.csv')


    run_script(prefix+"preprocess.py",issue_pr_df_path)
    # run_script(prefix+"LFR_network.py")

    projects = project_df['Repo'].unique()
    # 暂时使用其中一个仓库
    # projects = ['activemerchant/active_merchant']
    # 遍历项目列表中的每个仓库
    
    proj_df=pd.read_csv(f"{prefix}temp/project_list_filter.csv")
    projects=proj_df['proj'].tolist()
    count=0

    for proj in projects:
        print(f'============================  the {count}-th proj    ==============================')
        count+=1

        logging.info(f"Start processing project: {proj}")

        logging.info("running network_loader")
        run_script(prefix+"network_loader.py", proj)

        logging.info("Running non-overlapping algos")
        run_script(prefix+"non-overlapping.py", proj)

        logging.info("Running overlapping algos")
        run_script(prefix+"overlapping.py", proj)

        logging.info(f'Running metrics about graph topology')
        run_script(prefix+"metrics_topology.py",proj)

        logging.info("Running metrics for community detection")
        run_script(prefix+"main_metrics_community_detection.py", proj)

        logging.info("Running metrics for community with ground truth")
        run_script(prefix+"main_metrics_with_ground_truth.py", proj)

        logging.info("Running statistics for mined communties")
        run_script(prefix+"group_projects.py",proj)

    statastic_project_running(prefix)

    logging.info("Running LFR network exp")
    run_script(prefix+"LFR_network.py")
    run_script(prefix+"LFR_network_overlapping.py")



