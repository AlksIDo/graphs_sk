from loguru import logger

from models import DGLRegressor, NonGraphModel, PGRegressor
from utils import (
    baseline_model_pg_dataset,
    create_dgl_loaders,
    create_pg_loaders,
    dgl_model_train,
    non_graph_model_train,
    pg_model_train,
)

device = "cuda:2"


def main():
    (
        train_dataset_pg,
        test_dataset_pg,
        train_loader_pg,
        test_loader_pg,
    ) = create_pg_loaders(graph_dataset=True)

    (
        train_dataset_nong,
        test_dataset_nong,
        train_loader_nong,
        test_loader_nong,
    ) = create_pg_loaders(graph_dataset=False)

    (
        train_dataset_dgl,
        test_dataset_dgl,
        train_loader_dgl,
        test_loader_dgl,
    ) = create_dgl_loaders()

    # Baseline
    rmse_baseline, r2_baseline, torch_rmse = baseline_model_pg_dataset(
        train_dataset_pg, test_dataset_pg
    )
    logger.info(f"Baseline model results: RMSE on test dataset: {rmse_baseline}")

    # Non-graph model
    model = NonGraphModel()
    model = model.to(device)

    logger.info("Starting non-graph model train...")
    non_graph_model_rmse = []
    for epoch in range(201):
        rmse = non_graph_model_train(model, train_loader_nong, test_loader_nong, device)
        non_graph_model_rmse.append(rmse)
        if epoch % 50 == 0 and epoch != 0:
            logger.info(f"Epoch {epoch}. Min RMSE: {min(non_graph_model_rmse)}")
    logger.info(
        f"Non-graph model train finished! Min RMSE on test dataset: {min(non_graph_model_rmse)}"
    )

    # DGL graph model
    model = DGLRegressor(27, 120, 1).double()
    model = model.to(device)

    logger.info("Starting graph DGL model train...")
    dgl_graph_model_rmse = []
    for epoch in range(501):
        rmse = dgl_model_train(model, train_loader_dgl, test_loader_dgl, device)
        dgl_graph_model_rmse.append(rmse)
        if epoch % 50 == 0 and epoch != 0:
            logger.info(f"Epoch {epoch}. Min RMSE: {min(dgl_graph_model_rmse)}")
    logger.info(
        f"DGL graph model train finished! Min RMSE on test dataset: {min(dgl_graph_model_rmse)}"
    )

    # PG graph model
    model = PGRegressor(
        hidden_channels=120, node_feature_channels=train_dataset_pg.num_node_features
    )
    model = model.to(device)

    pg_graph_model_rmse = []
    for epoch in range(501):
        rmse = pg_model_train(model, train_loader_pg, test_loader_pg, device)
        pg_graph_model_rmse.append(rmse)
        if epoch % 50 == 0 and epoch != 0:
            logger.info(f"Epoch {epoch}. Min RMSE: {min(pg_graph_model_rmse)}")

    logger.info(
        f"Pytorch-geometric graph model train finished! Min RMSE on test dataset: {min(pg_graph_model_rmse)}"
    )

    logger.info(
        f"Comparison of models: BASELINE RMSE {rmse_baseline}, NON-GRAPH MODEL {min(non_graph_model_rmse)}, DGL GRAPH MIN RMSE {min(dgl_graph_model_rmse)}, PG GRAPH MIN RMSE {min(pg_graph_model_rmse)}"
    )

    return rmse_baseline, dgl_graph_model_rmse, pg_graph_model_rmse


if __name__ == "__main__":
    main()
