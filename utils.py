import os
from copy import deepcopy

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as OHE
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

MAX_NUM_NODES = 538


class NonGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_data = self.graphs[idx]

        graph_data.y_exp = torch.exp(deepcopy(graph_data.y))

        graph_data.y_int = torch.round(deepcopy(graph_data.y_exp)).float()

        return (
            F.pad(graph_data.x, (0, 0, MAX_NUM_NODES - graph_data.x.shape[0], 0)),
            graph_data.y,
        )


class GraphDataset(Dataset):
    def __init__(self, graphs, dgl_style=False):
        self.graphs = graphs
        self.dgl_style = dgl_style

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_data = self.graphs[idx]

        graph_data.y_exp = torch.exp(deepcopy(graph_data.y))

        graph_data.y_int = torch.round(deepcopy(graph_data.y_exp)).float()

        if not self.dgl_style:
            return graph_data
        else:
            return graph_data, graph_data.y


def create_pg_loaders(graph_dataset=True):
    """
    Create datasets and dataloaders for pytorch geometric model or for non-graph model
    """
    df = pd.read_excel("02-pdbbind-refined.xlsx", engine="openpyxl")
    df["e_predict"] = df["e_exp"] * 2 ** (-(df["rmsd"] ** 2) / 4)
    df["e_predict"] = df["e_predict"].map(lambda x: abs(x))

    folder = "data/"
    node_ohe = OHE(categories=[list(range(23)), [0, 1], [0, 1]])
    edge_ohe = OHE(categories=[[0, 1]])
    graphs = []
    idx_arr = []
    y_arr = []

    for num, fn in enumerate(os.listdir(folder)):
        if fn.endswith(".npz"):
            data = np.load(folder + fn)
            X = torch.tensor(
                node_ohe.fit_transform(data["node_data"][:, 1:4]).todense()
            )

            A = edge_ohe.fit_transform(data["edge_data"][:, 1:]).todense()
            E = torch.tensor(np.hstack((data["edge_data"][:, :1], A)))

            pg_data = Data(
                edge_index=torch.tensor(data["edges"][:, 0:2]).t().contiguous().long(),
                x=X.float(),
                edge_attr=E.float(),
                y=torch.tensor(
                    df[df["name"] == fn.replace(".npz", ".gml")]["e_predict"].item()
                ),
            )

            idx_arr.append(num)
            y_arr.append(
                df[df["name"] == fn.replace(".npz", ".gml")]["e_predict"].item()
            )

            graphs.append(pg_data)

    X_train, X_test, y_train, y_test = train_test_split(
        graphs,
        y_arr,
        test_size=0.2,
        random_state=42,
        stratify=[np.exp(i.y.cpu().numpy()) > 0.5 for i in graphs],
    )

    if graph_dataset:
        train_dataset = GraphDataset(X_train)
        test_dataset = GraphDataset(X_test)
    else:
        train_dataset = NonGraphDataset(X_train)
        test_dataset = NonGraphDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4)

    return train_dataset, test_dataset, train_loader, test_loader


def create_dgl_loaders():
    """
    Creates datasets and dataloaders for DGL model
    """
    df = pd.read_excel("02-pdbbind-refined.xlsx", engine="openpyxl")
    df["e_predict"] = df["e_exp"] * 2 ** (-(df["rmsd"] ** 2) / 4)
    df["e_predict"] = df["e_predict"].map(lambda x: abs(x))

    folder = "data/"
    node_ohe = OHE(categories=[list(range(23)), [0, 1], [0, 1]])
    edge_ohe = OHE(categories=[[0, 1]])
    graphs = []
    for fn in os.listdir(folder):
        if fn.endswith(".npz"):
            data = np.load(folder + fn)
            X = torch.tensor(
                node_ohe.fit_transform(data["node_data"][:, 1:4]).todense()
            )

            A = edge_ohe.fit_transform(data["edge_data"][:, 1:]).todense()
            E = torch.tensor(np.hstack((data["edge_data"][:, :1], A)))

            g = dgl.graph((data["edges"][:, 0], data["edges"][:, 1]))
            g.ndata["node_features"] = X
            g.edata["edge_features"] = E
            g.y = torch.tensor(
                df[df["name"] == fn.replace(".npz", ".gml")]["e_predict"].item()
            )
            g = dgl.add_self_loop(g)
            graphs.append(g)

    X_train, X_test, y_train, y_test = train_test_split(
        graphs,
        [i.y for i in graphs],
        test_size=0.2,
        random_state=42,
        stratify=[np.exp(i.y.cpu().numpy()) > 0.5 for i in graphs],
    )

    train_dataset = GraphDataset(X_train, dgl_style=True)
    test_dataset = GraphDataset(X_test, dgl_style=True)

    train_loader = GraphDataLoader(
        train_dataset, batch_size=10, drop_last=False, shuffle=True, num_workers=4
    )
    test_loader = GraphDataLoader(
        test_dataset, batch_size=10, drop_last=False, shuffle=True, num_workers=4
    )

    return train_dataset, test_dataset, train_loader, test_loader


def baseline_model_pg_dataset(train_dataset, test_dataset):
    """
    Create baseline model, which predict mean value on train_dataset, and measure RMSE
    """
    criterion = nn.MSELoss(reduction="mean")

    prediction = np.mean([i.y.item() for i in train_dataset])

    true_labels = [i.y.item() for i in test_dataset]
    pred_labels = [prediction for i in range(len(test_dataset))]

    return (
        mean_squared_error(pred_labels, true_labels, squared=True),
        r2_score(pred_labels, true_labels),
        criterion(torch.tensor(pred_labels), torch.tensor(true_labels)).item(),
    )


def non_graph_model_train(model, train_loader, test_loader, device):
    """
    Simple training pipeline for non-graph model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for data, y in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        y = y.to(device)
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out.view(-1, 1), y.view(-1, 1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    with torch.no_grad():
        true = []
        preds = []
        for data, y in test_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            y = y.to(device)
            out = model(data)  # Perform a single forward pass.
            # val_loss = criterion(out.view(-1, 1), y.view(-1, 1))  # Compute the loss.
            true.extend(y.detach().cpu().numpy())
            preds.extend(out.detach().cpu().numpy())

    return mean_squared_error(true, preds)


def dgl_model_train(model, train_loader, test_loader, device):
    """
    Simple training pipeline for dgl model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for batched_graph, labels in train_loader:
        feats = batched_graph.ndata["node_features"].to(device)
        logits = model(batched_graph.to(device), feats)
        loss = F.mse_loss(
            logits.double().view(1, -1), labels.double().view(1, -1).to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = []
    trues = []
    model.eval()
    for batched_graph, labels in test_loader:
        feats = batched_graph.ndata["node_features"].to(device)
        logits = model(batched_graph.to(device), feats)
        # val_loss = F.mse_loss(
        #     logits.double().view(1, -1), labels.double().view(1, -1).to(device)
        # )
        preds.extend(logits.detach().cpu().numpy())
        trues.extend(labels.detach().cpu().numpy())

    return mean_squared_error(preds, trues)


def pg_model_train(model, train_loader, test_loader, device):
    """
    Simple training pipeline for pytorch geometric model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out.view(-1, 1), data.y.view(-1, 1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    with torch.no_grad():
        true = []
        preds = []
        for data in test_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data)  # Perform a single forward pass.
            # val_loss = criterion(
            #     out.view(-1, 1), data.y.view(-1, 1)
            # )  # Compute the loss.
            true.extend(data.y.detach().cpu().numpy())
            preds.extend(out.detach().cpu().numpy())

    return mean_squared_error(true, preds)
