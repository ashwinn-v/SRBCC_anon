import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import normalized_mutual_info_score

from coclustering_vade import (
    ScaledVariationalCoClustering,
    CoclusteringDataset,
    pretrain_vae,
    initialize_gmm_parameters,
    ModelTrainer,
    plot_matrix,
    plot_rearranged_coclusters,
)

def run_coclustering_from_mat(mat_file_path: str):
    """
    Loads data from a .mat file containing:
      - 'fea' of shape (195, 1703)
      - 'gnd' of shape (195, 1)
    and runs the VaDE-based co-clustering pipeline.
    """

    # ---------------------------------------------------------------------
    # 1) Load data from .mat
    # ---------------------------------------------------------------------
    data_dict = scipy.io.loadmat(mat_file_path)

    # 'fea' -> shape (n_samples, n_features) = (195, 1703)
    # 'gnd' -> shape (n_samples, 1), flattened to shape (195,)
    X = data_dict['fea'].astype(np.float32)
    Y = data_dict['gnd'].flatten()

    print("X (fea) shape:", X.shape)   # expected (195, 1703)
    print("Y (gnd) shape:", Y.shape)   # expected (195,)

    # ---------------------------------------------------------------------
    # 2) Convert to torch Tensors
    # ---------------------------------------------------------------------
    data_tensor = torch.tensor(X, dtype=torch.float32)  # shape: (195, 1703)
    row_labels = Y
    col_labels = None  # This dataset has no column labels

    # ---------------------------------------------------------------------
    # 3) Create the CoclusteringDataset
    # ---------------------------------------------------------------------
    # Treat the 195 samples as "rows" and the 1703 features as "columns".
    dataset = CoclusteringDataset(
        data_matrix=data_tensor,
        batch_size=32,       # adjust if needed
        normalize=True,
        row_labels=row_labels,
        col_labels=col_labels
    )

    # Optional: plot the raw matrix (uncomment if needed)
    # plot_matrix(X, title="Raw Feature Matrix", filename="raw_fea_matrix.png")

    # ---------------------------------------------------------------------
    # 4) Build the co-clustering model
    # ---------------------------------------------------------------------
    row_input_dim = 1703  # number of features
    col_input_dim = 195   # number of samples
    row_components = 10   # number of row-side clusters
    col_components = 10   # number of column-side clusters

    # Example hidden dimensions and latent dimensions (tweak as necessary)
    row_hidden_dims = [512, 256]
    col_hidden_dims = [128, 64]
    row_latent_dim = 32
    col_latent_dim = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ScaledVariationalCoClustering(
        row_input_dim=row_input_dim,
        row_hidden_dims=row_hidden_dims,
        row_latent_dim=row_latent_dim,
        row_components=row_components,
        col_input_dim=col_input_dim,
        col_hidden_dims=col_hidden_dims,
        col_latent_dim=col_latent_dim,
        col_components=col_components,
        lambda_mi=1.0,  # mutual-information weight
        device=device
    )

    # ---------------------------------------------------------------------
    # 5) Pre-train the row VAE (standard Gaussian prior)
    # ---------------------------------------------------------------------
    print("\n[Pretrain] Row-side VAE (features):")
    pretrain_vae(
        vae_model=model.row_model,
        data_loader=dataset.row_loader,
        device=device,
        n_pretrain_epochs=20,
        learning_rate=1e-3
    )

    print("[Initialize GMM] Row side:")
    initialize_gmm_parameters(model.row_model, dataset.row_loader, device)

    # ---------------------------------------------------------------------
    # 6) Pre-train the column VAE (samples)
    # ---------------------------------------------------------------------
    print("\n[Pretrain] Column-side VAE (samples):")
    pretrain_vae(
        vae_model=model.col_model,
        data_loader=dataset.col_loader,
        device=device,
        n_pretrain_epochs=20,
        learning_rate=1e-3
    )

    print("[Initialize GMM] Column side:")
    initialize_gmm_parameters(model.col_model, dataset.col_loader, device)

    # Optional: initial co-cluster arrangement
    row_clusters_init, col_clusters_init = model.get_cluster_assignments(
        dataset.row_data.to(device),
        dataset.col_data.to(device)
    )
    # Uncomment to plot:
    # plot_rearranged_coclusters(
    #     data_matrix=data_tensor.cpu().numpy(),
    #     row_clusters=row_clusters_init.cpu().numpy(),
    #     col_clusters=col_clusters_init.cpu().numpy(),
    #     title="Initial Co-clustering",
    #     filename="initial_coclust.png"
    # )

    # ---------------------------------------------------------------------
    # 7) Train with KL-annealing + mutual-info weighting
    # ---------------------------------------------------------------------
    beta_schedule = [0.1]*5 + [0.5]*5 + [1.0]*10

    trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        learning_rate=1e-3,
        beta_schedule=beta_schedule,
        recon_weight=1.0,
        kl_weight=1.0,
        mi_weight=20.0
    )

    print("\n[Training] Co-clustering on the 195×1703 dataset...")
    results = trainer.train(
        n_epochs=40,
        evaluate_every=5,
        early_stopping_patience=5
    )

    # ---------------------------------------------------------------------
    # 8) Final evaluation + optional plot
    # ---------------------------------------------------------------------
    final_eval = trainer.evaluate()
    row_clusters_final = final_eval['row_clusters']
    col_clusters_final = final_eval['col_clusters']

    print("\n[Results] Final MI loss:", final_eval['mutual_info_loss'])
    if final_eval['row_accuracy'] is not None:
        print("Row Clustering Accuracy:", final_eval['row_accuracy'])
    if final_eval['row_nmi'] is not None:
        print("Row NMI:", final_eval['row_nmi'])
    if final_eval['col_accuracy'] is not None:
        print("Column Clustering Accuracy:", final_eval['col_accuracy'])
    if final_eval['col_nmi'] is not None:
        print("Column NMI:", final_eval['col_nmi'])

    # Final co-cluster arrangement plot
    plot_rearranged_coclusters(
        data_matrix=data_tensor.cpu().numpy(),
        row_clusters=row_clusters_final.cpu().numpy(),
        col_clusters=col_clusters_final.cpu().numpy(),
        title="Final Co-clustering",
        filename="final_coclust.png"
    )

if __name__ == "__main__":
    mat_path = "path/to/WebKB_cornell.mat"  # Update to your .mat file location
    run_coclustering_from_mat(mat_path)
