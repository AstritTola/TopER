
# TopER: Topological Embeddings in Graph Representation Learning

## Introduction
This repository provides the official implementation of **TopER (Topological Evolution Rate)**, introduced in our NeurIPS 2025 paper:  
https://openreview.net/pdf?id=JjTLq7SXaB.

TopER is a **novel, topology-inspired graph embedding method** that combines interpretability, scalability, and efficiency. Unlike traditional graph neural networks or persistent homology approaches, TopER directly summarizes the **filtration process** of a graph into **low-dimensional, interpretable embeddings**. These embeddings enable:

- **Intuitive Visualization**: 2D representations that reveal clusters, outliers, and structural patterns across graph datasets.
- **Competitive Performance**: State-of-the-art results in graph classification and clustering tasks on molecular, biological, and social network datasets.
- **Scalability**: Handles large graphs (100K+ nodes) in minutes, avoiding the cubic complexity of persistent homology.

### Key Features
- **Topology-Inspired Design**: Captures structural evolution without computing costly persistence diagrams.
- **Low-Dimensional Embeddings**: Pivot–growth descriptors for easy visualization and interpretation.
- **Robust & Efficient**: Stable under perturbations and practical for large-scale applications.



## Citation
If you use this code or method in your research, please cite:

```bibtex
@inproceedings{tola2025toper,
  title     = {TopER: Topological Embeddings in Graph Representation Learning},
  author    = {Astrit Tola and Funmilola Mary Taiwo and Cuneyt Gurcan Akcora and Baris Coskunuzer},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/pdf?id=JjTLq7SXaB}
}
```


## Installation
Install TopER using pip:

```bash
pip install toper
```



# Example: Using TopER with Datasets Folder

> **Note:** The `graph_processing` function shown below will be included in **TopER v2.0**.

## Example Code

```python
import toper
import pandas as pd
import numpy as np
import networkx as nx

def graph_processing(num_graph: int, edgedata: pd.DataFrame, graph_ind: list[int]):
    """
    Build a list of NetworkX graphs from edge data and node graph indices.
    
    Parameters
    ----------
    num_graph : int
        Total number of graphs.
    edgedata : pd.DataFrame
        Must contain columns ['from', 'to'] describing edges.
    graph_ind : list[int]
        A list where each entry i corresponds to the graph ID of node i+1.
        
    Returns
    -------
    list[nx.Graph]
        List of NetworkX Graph objects.
    """
    node_to_graph = pd.Series(graph_ind, index=np.arange(1, len(graph_ind) + 1))
    edge_graph_ids = node_to_graph.loc[edgedata['from']].values

    graphs = []
    for graph_id in range(1, num_graph + 1):
        mask = edge_graph_ids == graph_id
        sub_edges = edgedata.loc[mask, ['from', 'to']].to_numpy()

        G = nx.Graph()
        G.add_edges_from(sub_edges)

        graph_nodes = np.where(node_to_graph.values == graph_id)[0] + 1
        G.add_nodes_from(graph_nodes)

        graphs.append(G)

    return graphs

def load_data(name):
    path = f"<path_to_file>/{name}/"
    edgedata = pd.read_csv(path + f"{name}_A.txt", header=None, names=['from', 'to'])
    graph_ind = pd.read_csv(path + f"{name}_graph_indicator.txt", header=None).values.flatten().astype(int)
    return edgedata, graph_ind

# Example usage
name = 'BZR'
edgedata, graph_ind = load_data(name)
num_graph = len(np.unique(graph_ind))

van_graph_list = graph_processing(num_graph, edgedata, graph_ind)

# Compute TopER embeddings using degree centrality with 501 thresholds
F = toper.main(van_graph_list, ['deg_cen'], num_segm=501)
```

# Example: Using TopER with BZR from PyTorch Geometric

> **Note:** The `graph_processing_torch` function shown below will be included in **TopER v2.0**.


---

## Code

```python
import toper
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset

def graph_processing_torch(dataset):
    """
    Convert a PyTorch Geometric dataset into a list of NetworkX graphs.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        A PyG dataset where each element is a graph with edge_index and num_nodes.

    Returns
    -------
    list[nx.Graph]
        List of NetworkX Graph objects.
    """
    graphs = []
    for data in dataset:
        # Create a NetworkX graph for each PyG graph
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        edges = [(int(u), int(v)) for u, v in zip(edge_index[0], edge_index[1])]
        G.add_edges_from(edges)

        # Add isolated nodes if any
        G.add_nodes_from(range(data.num_nodes))

        graphs.append(G)
    return graphs


root = "./data/TUDataset"
dataset_name = "BZR"
pyg_dataset = TUDataset(root=root, name=dataset_name)

van_graph_list = graph_processing_torch(pyg_dataset)

# Example: compute TopER embeddings using degree centrality with 501 thresholds
F = toper.main(van_graph_list, ['deg_cen'], num_segm=501)
```






