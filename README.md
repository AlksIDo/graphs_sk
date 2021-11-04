# Graph Neural Networks for energy prediction

Here are results of my work

## Notebooks
Experiments.ipynb contains several high-level models, utilizing edge features, which showed low performance. It could be due to LR problems, bugged work of mixed precision with pytorch-geometric and dgl, etc.

Working_models.ipynb contains working pytorch-geometric and dgl models.

## Usage of pipeline
Pipeline were tested on python 3.9.1 version

```bash
python3 main.py
```

Results will be outputted as logs, in file run_example.txt you could check results of a single run.

Latest result:
```python
2021-11-03 22:18:22.138 | INFO     | __main__:main:89 - Comparison of models: BASELINE RMSE 8.563990328446478, NON-GRAPH MODEL 4.17226505279541, DGL GRAPH MIN RMSE 1.8091076462585127, PG GRAPH MIN RMSE 2.3979108333587646
```
