This project is the data and source code for the paper "Benchmarking Community Detection for Open Source Software Developers' Communication Networks"

# Usage

**Installing requirements**: This project provides a requirement. txt file, which can be quickly installed with the following command to obtain the necessary packages for the project:

```
cd benchmark_community_detection_in_OSS/
pip install -r requirements.txt
```

**Running project**: The project can be executed using the following command:

```
python main.py
```

- If it prompts that a folder does not exist, the "prefix" configuration of the file needs to be modified to the project path corresponding to the running device

- The `output/` folder stores the output of all code, including the corresponding metrics for each project and the statistical charts generated based on the metrics. The `temp/` folder stores the intermediate files.

# Project Structure

**Execution Process**: The execution process mainly follows the steps below.

- **Preprocess**: The raw data is stored in the `data-origin/` directory, and `preprocess.py` reads the contents of the folder. After removing the bot and filtering the data, the data is split by project and stored in the `data/` as a CSV file for each project, named in the format of `[username]@[repo name].csv`.

- **Generate Synthetic Networks** with OSS DSN Characteristics: Run `LFR_network.py` to generate different LFR benchmark networks and compare the performance of different algorithms. The LFR network is stored in `data/lfr_network`.

For each network `data/`:

- **Establish OSS-DSNs**: Read the corresponding CSV file of the project in `network_roader.py`, create a weighted undirected graph for the project, create a new directory and save it in `temp/graphs/[username]@[reponame]-graph.graphml`
- **Community mining**: Using multiple non-overlapping and overlapping community detection algorithms for DSN and synthetic networks. All algorithm implementations are stored in `non-overlapping.py` and `overlapping.py`, and the communities mined by different algorithms are ultimately stored in `temp/communities/[username]@[reponame]-[algorithm name].pkl`.
- **Calculation of the metrics** listed in the article: The calculation is stored in `metricsNonOverlapping.py`, `metricsNonOverlappingWithGroundTruth.py`, `metricsOverlapping.py`, `metricsOverlappingWithGroundTruth.py`.The metrics of each project are stored in `output/graph-metrics/[username]@[reponame].json`
- **Visualization** in the article: All charts and table in the paper are generated by `draw_metric_pic.py` and `visualization.py`.