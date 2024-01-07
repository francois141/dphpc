# Python benchmarking instructions

- Install Pytorch (2.1.2 is not supported by DGL yet so install at most 2.1.1)
- [Install DGL from conda or pip](https://docs.dgl.ai/en/1.1.x/install/index.html#install-from-conda-or-pip)
- Run `python python/benchmark_dgl.py` and specify the data folder, K and num_runs values
- The results are printed to the console and can be written to a file with `>`, e.g. `python python/becnhmark_dgl.py --data_folder ../data > result.csv`
