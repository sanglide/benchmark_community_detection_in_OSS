This project is the data and source code for the paper "Benchmarking Community Detection for Open Source Software Developers' Communication Networks"

The code mainly includes the following functions:
- Establish OSS-DSNs from project commit, issue, and PR data
- Community mining using multiple non-overlapping and overlapping community detection algorithms for DSN and synthetic networks
- Generation of Synthetic Networks with OSS DSN Characteristics
- Calculation of the indicators listed in the article
- Visualization content in the article

The precautions for running code are as follows:

- The project running entrance is in main.py
  - If it prompts that a folder does not exist, the "prefix" configuration of the file needs to be modified to the project path corresponding to the running device
- -The results are stored in the "outputs/", and the intermediate files are in the "temp/"