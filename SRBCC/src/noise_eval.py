import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.datasets import make_checkerboard
from typing import List, Tuple, Optional
import logging

##############################################################################
# 1) Compute responsibilities using learnable mixture weights
##############################################################################
def compute_mixture_responsibilities(
    z: torch.Tensor,
    log_weights: torch.Tensor,
    means: torch.Tensor,
    covs: torch.Tensor
) -> torch.Tensor:
    """
    Compute responsibilities (gamma) for GMM components using the latent sample z,
    with learnable mixture weights (via log-weights + softmax), means, and covs.
    """
    batch_size = z.size(0)
    n_components = means.size(0)
    latent_dim = z.size(1)

    # Expand for broadcasting
    z = z.unsqueeze(1)         # (batch_size, 1, latent_dim)
    means = means.unsqueeze(0) # (1, n_components, latent_dim)
    covs  = covs.unsqueeze(0)  # (1, n_components, latent_dim)

    # Convert log_weights to normalized mixture weights
    log_pi_c = F.log_softmax(log_weights, dim=0)  # shape: (n_components,)

    # log prob under each Gaussian:
    # -0.5 * [sum_j( log(2*pi*cov_j) + (z_j - mu_j)^2 / cov_j )]
    log_p = -0.5 * (
        torch.sum(torch.log(2.0 * np.pi * covs), dim=2) +
        torch.sum((z - means)**2 / covs, dim=2)
    )  # shape: (batch_size, n_components)

    # Add the mixture log-weights
    log_p = log_p + log_pi_c  # shape: (batch_size, n_components)

    # Normalize via log-sum-exp to get log responsibilities
    log_gamma = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    gamma = torch.exp(log_gamma)  # shape: (batch_size, n_components)

    return gamma


##############################################################################
# 2) VAE_GMM with learnable log-weights, means, and covs
##############################################################################
class VAE_GMM(nn.Module):
    """
    Variational Autoencoder with Gaussian Mixture prior (VaDE-style).
    Includes a learnable mixture-weight parameter.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 n_components: int):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
            
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # GMM parameters (learnable)
        self.n_components = n_components
        self.latent_dim = latent_dim

        self.gmm_log_weights = nn.Parameter(torch.zeros(n_components))
        self.gmm_means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.gmm_log_covs = nn.Parameter(torch.zeros(n_components, latent_dim))  # log variances

    def get_positive_covs(self) -> torch.Tensor:
        return F.softplus(self.gmm_log_covs)  # ensures positivity

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_layers(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def get_responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        covs = self.get_positive_covs()
        return compute_mixture_responsibilities(z, self.gmm_log_weights, self.gmm_means, covs)
    
    def loss_function(self, 
                      x: torch.Tensor,
                      x_recon: torch.Tensor,
                      mu: torch.Tensor,
                      logvar: torch.Tensor,
                      z: torch.Tensor,
                      beta: float = 1.0,
                      recon_weight: float = 1.0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, d = mu.shape
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        covs = self.get_positive_covs()  # (n_components, latent_dim)
        gamma = self.get_responsibilities(z)  # (N, n_components)
        
        # Broadcast shapes
        mu_exp = mu.unsqueeze(1)         # (N, 1, d)
        logvar_exp = logvar.unsqueeze(1) # (N, 1, d)
        means = self.gmm_means.unsqueeze(0)  # (1, n_components, d)
        covs_expanded = covs.unsqueeze(0)    # (1, n_components, d)

        # Mixture term in KL
        constant = d * np.log(2.0 * np.pi)
        mixture_term = 0.5 * (
            constant
            + torch.sum(torch.log(covs_expanded), dim=2)
            + torch.sum(torch.exp(logvar_exp) / covs_expanded, dim=2)
            + torch.sum((mu_exp - means)**2 / covs_expanded, dim=2)
        )
        kl_mixture = torch.sum(gamma * mixture_term, dim=1)  # (N,)

        # Standard VAE KL part
        kl_standard = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (N,)

        # Mixture-weight log p(c)
        log_pi = F.log_softmax(self.gmm_log_weights, dim=0)  # (n_components,)
        kl_prior = - torch.sum(gamma * log_pi, dim=1)  # (N,)

        # Entropy term
        kl_entropy = torch.sum(gamma * torch.log(gamma + 1e-8), dim=1)

        kl_per_sample = kl_mixture + kl_standard + kl_prior + kl_entropy
        kl_loss = torch.sum(kl_per_sample)

        total_loss = recon_weight * recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


##############################################################################
# 3) The co-clustering system with VaDE
##############################################################################
class ScaledVariationalCoClustering:
    """
    Co-clustering model with VaDE-based VAE_GMM for both row and column sides,
    plus a mutual-information coupling term.
    """
    def __init__(self, 
                 row_input_dim: int,
                 row_hidden_dims: List[int],
                 row_latent_dim: int,
                 row_components: int,
                 col_input_dim: int,
                 col_hidden_dims: List[int],
                 col_latent_dim: int,
                 col_components: int,
                 lambda_mi: float = 1.0,
                 device: str = 'cpu'):
        self.device = device
        
        self.row_model = VAE_GMM(
            row_input_dim, row_hidden_dims, 
            row_latent_dim, row_components
        ).to(device)
        
        self.col_model = VAE_GMM(
            col_input_dim, col_hidden_dims, 
            col_latent_dim, col_components
        ).to(device)
        
        self.lambda_mi = lambda_mi
    
    def train_step(self, 
                   row_data_batch: torch.Tensor,
                   col_data_batch: torch.Tensor,
                   data_matrix: torch.Tensor,
                   optimizer_row: torch.optim.Optimizer,
                   optimizer_col: torch.optim.Optimizer,
                   beta: float = 1.0,
                   recon_weight: float = 1.0,
                   kl_weight: float = 1.0,
                   mi_weight: float = 1.0
                  ) -> Tuple[float, float, float]:
        optimizer_row.zero_grad()
        x_recon, mu_q, logvar_q, z = self.row_model(row_data_batch)
        loss_row, _, _ = self.row_model.loss_function(
            row_data_batch, x_recon, mu_q, logvar_q, z,
            beta=beta * kl_weight, recon_weight=recon_weight
        )
        
        optimizer_col.zero_grad()
        y_recon, mu_c, logvar_c, z_c = self.col_model(col_data_batch)
        loss_col, _, _ = self.col_model.loss_function(
            col_data_batch, y_recon, mu_c, logvar_c, z_c,
            beta=beta * kl_weight, recon_weight=recon_weight
        )
        
        row_mu, _ = self.row_model.encoder(row_data_batch)
        col_mu, _ = self.col_model.encoder(col_data_batch)
        gamma_rows = self.row_model.get_responsibilities(row_mu)
        gamma_cols = self.col_model.get_responsibilities(col_mu)

        cross_loss = mi_weight * mutual_info_co_clustering_loss(
            gamma_rows, gamma_cols, data_matrix, self.lambda_mi
        )
        
        total_loss = loss_row + loss_col + cross_loss
        total_loss.backward()
        optimizer_row.step()
        optimizer_col.step()
        
        return loss_row.item(), loss_col.item(), cross_loss.item()
    
    def get_cluster_assignments(self, 
                                row_data: torch.Tensor,
                                col_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.row_model.eval()
        self.col_model.eval()
        
        with torch.no_grad():
            row_mu, _ = self.row_model.encoder(row_data)
            col_mu, _ = self.col_model.encoder(col_data)
            
            gamma_rows = self.row_model.get_responsibilities(row_mu)
            gamma_cols = self.col_model.get_responsibilities(col_mu)
            
            row_clusters = gamma_rows.argmax(dim=1)
            col_clusters = gamma_cols.argmax(dim=1)
            
        return row_clusters, col_clusters
    
    def save_model(self, path: str):
        torch.save({
            'row_model_state': self.row_model.state_dict(),
            'col_model_state': self.col_model.state_dict(),
            'lambda_mi': self.lambda_mi
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.row_model.load_state_dict(checkpoint['row_model_state'])
        self.col_model.load_state_dict(checkpoint['col_model_state'])
        self.lambda_mi = checkpoint['lambda_mi']


##############################################################################
# Pre-training function
##############################################################################
def pretrain_vae(
    vae_model: VAE_GMM, 
    data_loader: DataLoader, 
    device: torch.device, 
    n_pretrain_epochs: int = 10, 
    learning_rate: float = 1e-3
):
    """
    Pre-train a VAE model using a standard VAE objective (recon + KL to N(0,I)).
    """
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    vae_model.train()
    for epoch in range(n_pretrain_epochs):
        total_loss = 0.0
        count = 0
        for (batch,) in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = vae_model(batch)
            
            recon_loss = F.mse_loss(x_recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += batch.size(0)
        print(f"[Pretrain VAE] Epoch {epoch+1}/{n_pretrain_epochs}, "
              f"Loss per sample: {total_loss / count:.4f}")


##############################################################################
# 4) Data preprocessing / dataset
##############################################################################
class DataPreprocessor:
    """
    Handles data preprocessing for co-clustering, including normalization.
    """
    def __init__(self, normalize_rows: bool = True, normalize_cols: bool = True):
        self.normalize_rows = normalize_rows
        self.normalize_cols = normalize_cols
        self.row_mean = None
        self.row_std = None
        self.col_mean = None
        self.col_std = None
    
    def fit_transform(self, data_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_rows:
            self.row_mean = data_matrix.mean(dim=1, keepdim=True)
            self.row_std  = data_matrix.std(dim=1, keepdim=True) + 1e-8
            row_data = (data_matrix - self.row_mean) / self.row_std
        else:
            row_data = data_matrix
            
        if self.normalize_cols:
            self.col_mean = data_matrix.mean(dim=0, keepdim=True)
            self.col_std  = data_matrix.std(dim=0, keepdim=True) + 1e-8
            col_data = (data_matrix - self.col_mean) / self.col_std
        else:
            col_data = data_matrix
            
        return row_data, col_data.t()


class CoclusteringDataset:
    """
    Dataset wrapper for co-clustering that handles row and column data
    and optional true labels.
    """
    def __init__(self, 
                 data_matrix: torch.Tensor,
                 batch_size: int = 128,
                 normalize: bool = True,
                 row_labels: Optional[np.ndarray] = None,
                 col_labels: Optional[np.ndarray] = None):
        self.preprocessor = DataPreprocessor(normalize_rows=normalize, 
                                             normalize_cols=normalize)
        self.data_matrix = data_matrix
        
        self.row_labels = row_labels
        self.col_labels = col_labels
        
        row_data, col_data = self.preprocessor.fit_transform(data_matrix)
        self.row_data = row_data
        self.col_data = col_data
        
        self.batch_size = batch_size
        
        self.row_dataset = TensorDataset(self.row_data)
        self.col_dataset = TensorDataset(self.col_data)
        
        self.row_loader = DataLoader(self.row_dataset, batch_size=batch_size, shuffle=True)
        self.col_loader = DataLoader(self.col_dataset, batch_size=batch_size, shuffle=True)


##############################################################################
# 5) MI-based cross-loss
##############################################################################
def mutual_info_co_clustering_loss(
    gamma_rows: torch.Tensor, 
    gamma_cols: torch.Tensor,
    data_matrix: torch.Tensor,
    lambda_mi: float
) -> torch.Tensor:
    device = gamma_rows.device
    eps = 1e-12

    B = min(gamma_rows.size(0), gamma_cols.size(0))
    gamma_rows = gamma_rows[:B]
    gamma_cols = gamma_cols[:B]

    T_pro_org = torch.matmul(gamma_rows, gamma_cols.t())
    T_pro_org = T_pro_org / (T_pro_org.sum() + eps)

    row_idx = gamma_rows.argmax(dim=1)
    col_idx = gamma_cols.argmax(dim=1)
    n_row_clusters = gamma_rows.size(1)
    n_col_clusters = gamma_cols.size(1)

    I = row_idx.unsqueeze(1).expand(-1, B).reshape(-1)
    J = col_idx.unsqueeze(0).expand(B, -1).reshape(-1)
    values = T_pro_org.reshape(-1)

    T_pro_red_flat = torch.zeros(n_row_clusters * n_col_clusters,
                                 device=device,
                                 dtype=gamma_rows.dtype)
    target_indices = I * n_col_clusters + J
    T_pro_red_flat = T_pro_red_flat.scatter_add(0, target_indices, values)
    T_pro_red = T_pro_red_flat.view(n_row_clusters, n_col_clusters)
    T_pro_red = T_pro_red / (T_pro_red.sum() + eps)

    def compute_mi(table: torch.Tensor) -> torch.Tensor:
        pr = table.sum(dim=1, keepdim=True)
        pc = table.sum(dim=0, keepdim=True)
        ratio = (table + eps) / ((pr @ pc) + eps)
        return (table * torch.log2(ratio + eps)).sum()
    
    MI_org = compute_mi(T_pro_org)
    MI_red = compute_mi(T_pro_red)
    
    ratio = MI_red / (MI_org + eps)
    loss_value = torch.log(1.0 + torch.abs(1.0 - ratio))
    
    return lambda_mi * loss_value


##############################################################################
# 6) Clustering Accuracy
##############################################################################
def clustering_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    true_labels = np.asarray(true_labels).flatten()
    pred_labels = np.asarray(pred_labels).flatten()
    n_samples = len(true_labels)
    
    true_labels = true_labels.astype(np.int64)
    pred_labels = pred_labels.astype(np.int64)
    
    unique_pred_clusters = np.unique(pred_labels)
    correct = 0
    for c in unique_pred_clusters:
        idx = np.where(pred_labels == c)[0]
        majority_label = np.bincount(true_labels[idx]).argmax()
        correct += (true_labels[idx] == majority_label).sum()
    return correct / n_samples


##############################################################################
# 7) Training loop
##############################################################################
class ModelTrainer:
    def __init__(self,
                 model: ScaledVariationalCoClustering,
                 dataset: CoclusteringDataset,
                 learning_rate: float = 1e-3,
                 beta_schedule: Optional[List[float]] = None,
                 recon_weight: float = 1.0,
                 kl_weight: float = 1.0,
                 mi_weight: float = 1.0):
        self.model = model
        self.dataset = dataset
        self.beta_schedule = beta_schedule or [1.0]
        
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.mi_weight = mi_weight
        
        self.optimizer_row = optim.Adam(model.row_model.parameters(), lr=learning_rate)
        self.optimizer_col = optim.Adam(model.col_model.parameters(), lr=learning_rate)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.row_model.train()
        self.model.col_model.train()
        
        total_row_loss = 0.0
        total_col_loss = 0.0
        total_cross_loss = 0.0
        n_batches = 0
        
        beta = self.beta_schedule[min(epoch, len(self.beta_schedule) - 1)]
        
        for (row_batch,), (col_batch,) in zip(self.dataset.row_loader, self.dataset.col_loader):
            row_batch = row_batch.to(self.model.device)
            col_batch = col_batch.to(self.model.device)
            
            row_loss, col_loss, cross_loss = self.model.train_step(
                row_batch, 
                col_batch,
                self.dataset.data_matrix.to(self.model.device),
                self.optimizer_row,
                self.optimizer_col,
                beta=beta,
                recon_weight=self.recon_weight,
                kl_weight=self.kl_weight,
                mi_weight=self.mi_weight
            )
            total_row_loss += row_loss
            total_col_loss += col_loss
            total_cross_loss += cross_loss
            n_batches += 1
            
        return (
            total_row_loss / n_batches,
            total_col_loss / n_batches,
            total_cross_loss / n_batches
        )
    
    def evaluate(self) -> dict:
        self.model.row_model.eval()
        self.model.col_model.eval()
        
        with torch.no_grad():
            row_data = self.dataset.row_data.to(self.model.device)
            col_data = self.dataset.col_data.to(self.model.device)
            row_clusters, col_clusters = self.model.get_cluster_assignments(row_data, col_data)
            
            row_mu, _ = self.model.row_model.encoder(row_data)
            col_mu, _ = self.model.col_model.encoder(col_data)
            gamma_rows = self.model.row_model.get_responsibilities(row_mu)
            gamma_cols = self.model.col_model.get_responsibilities(col_mu)
            
            mi_loss = mutual_info_co_clustering_loss(
                gamma_rows, gamma_cols, 
                self.dataset.data_matrix.to(self.model.device),
                self.model.lambda_mi
            )
            
            row_acc = None
            row_nmi = None
            col_acc = None
            col_nmi = None
            if self.dataset.row_labels is not None:
                row_acc = clustering_accuracy(self.dataset.row_labels, row_clusters.cpu().numpy())
                row_nmi = normalized_mutual_info_score(self.dataset.row_labels, row_clusters.cpu().numpy())
            if self.dataset.col_labels is not None:
                col_acc = clustering_accuracy(self.dataset.col_labels, col_clusters.cpu().numpy())
                col_nmi = normalized_mutual_info_score(self.dataset.col_labels, col_clusters.cpu().numpy())
            
        return {
            'row_clusters': row_clusters,
            'col_clusters': col_clusters,
            'mutual_info_loss': mi_loss.item(),
            'gamma_rows': gamma_rows,
            'gamma_cols': gamma_cols,
            'row_accuracy': row_acc,
            'row_nmi': row_nmi,
            'col_accuracy': col_acc,
            'col_nmi': col_nmi
        }
    
    def train(self, 
              n_epochs: int,
              evaluate_every: int = 1,
              early_stopping_patience: Optional[int] = None) -> dict:
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(n_epochs):
            row_loss, col_loss, cross_loss = self.train_epoch(epoch)
            total_loss = row_loss + col_loss + cross_loss
            
            self.logger.info(
                f"Epoch {epoch+1}/{n_epochs}: row_loss={row_loss:.4f}, "
                f"col_loss={col_loss:.4f}, cross_loss={cross_loss:.4f}"
            )
            
            training_history.append({
                'epoch': epoch,
                'row_loss': row_loss,
                'col_loss': col_loss,
                'cross_loss': cross_loss,
                'total_loss': total_loss
            })
            
            if (epoch + 1) % evaluate_every == 0:
                eval_metrics = self.evaluate()
                self.logger.info(
                    "Evaluation -> MI loss: {:.4f}, row_acc: {}, row_nmi: {}, "
                    "col_acc: {}, col_nmi: {}".format(
                        eval_metrics['mutual_info_loss'],
                        eval_metrics['row_accuracy'],
                        eval_metrics['row_nmi'],
                        eval_metrics['col_accuracy'],
                        eval_metrics['col_nmi'],
                    )
                )
                if early_stopping_patience:
                    if total_loss < best_loss:
                        best_loss = total_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info("Early stopping triggered!")
                        break
        
        final_eval = self.evaluate()
        return {
            'training_history': training_history,
            'final_evaluation': final_eval
        }


##############################################################################
# Helper: Plot rearranged co-clusters
##############################################################################
def plot_rearranged_coclusters(data_matrix: np.ndarray, row_clusters: np.ndarray, col_clusters: np.ndarray, title: str, filename: str):
    row_order = np.argsort(row_clusters)
    col_order = np.argsort(col_clusters)
    rearranged = data_matrix[row_order, :][:, col_order]
    plt.figure(figsize=(6, 5))
    plt.imshow(rearranged, aspect='auto', cmap='plasma')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_matrix(data_matrix: np.ndarray, title: str, filename: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(data_matrix, aspect='auto', cmap='plasma')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


##############################################################################
# 9) K-means Initialization for GMM
##############################################################################
def initialize_gmm_parameters(vae_model: VAE_GMM, data_loader: DataLoader, device: torch.device):
    latent_codes = []
    vae_model.eval()
    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            _, mu, _, _ = vae_model(batch)
            latent_codes.append(mu.cpu().numpy())
    latent_codes = np.concatenate(latent_codes, axis=0)
    
    kmeans = KMeans(n_clusters=vae_model.n_components, n_init=10)
    cluster_assignments = kmeans.fit_predict(latent_codes)
    centers = kmeans.cluster_centers_
    
    vae_model.gmm_means.data = torch.tensor(centers, dtype=vae_model.gmm_means.data.dtype, device=device)
    
    covs = []
    for i in range(vae_model.n_components):
        codes_i = latent_codes[cluster_assignments == i]
        if len(codes_i) > 0:
            cov = np.var(codes_i, axis=0) + 1e-6
        else:
            cov = np.ones(vae_model.latent_dim)
        covs.append(cov)
    covs = np.stack(covs, axis=0)
    vae_model.gmm_log_covs.data = torch.log(torch.tensor(covs, dtype=vae_model.gmm_log_covs.data.dtype, device=device))
    
    counts = np.bincount(cluster_assignments, minlength=vae_model.n_components)
    weights = counts / float(len(latent_codes))
    vae_model.gmm_log_weights.data = torch.tensor(np.log(weights + 1e-6), dtype=vae_model.gmm_log_weights.data.dtype, device=device)
    print("GMM parameters initialized via k-means.")


##############################################################################
# 10) Example main usage
##############################################################################
def main():
    """
    Demo: VaDE-based co-clustering on a (1000 x 1000) checkerboard with multiple noise levels.
    """
    shape = (1000, 1000)
    n_clusters = (4, 4)  # row x col

    # Generate clean checkerboard data
    data_clean, row_bools, col_bools = make_checkerboard(
        shape=shape,
        n_clusters=n_clusters,
        noise=0.0,
        shuffle=False,
        random_state=42
    )
    row_labels_true_clean = np.argmax(row_bools, axis=0)
    col_labels_true_clean = np.argmax(col_bools, axis=0)
    data_std = data_clean.std()

    noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    final_results = []

    for noise_frac in noise_levels:
        rng = np.random.RandomState(0)
        noise_level = noise_frac * data_std
        noise = rng.randn(*data_clean.shape) * noise_level
        data_noisy = data_clean + noise

        row_idx_shuffled = rng.permutation(data_noisy.shape[0])
        col_idx_shuffled = rng.permutation(data_noisy.shape[1])
        data_shuffled = data_noisy[row_idx_shuffled][:, col_idx_shuffled]

        row_labels_true = row_labels_true_clean[row_idx_shuffled]
        col_labels_true = col_labels_true_clean[col_idx_shuffled]

        input_plot_filename = f"input_data_{int(noise_frac*100)}percent_noise_shuffled.png"
        plot_matrix(
            data_matrix=data_shuffled,
            title=f"Input Data (Shuffled, {int(noise_frac*100)}% Noise)",
            filename=input_plot_filename
        )
        print(f"Saved: {input_plot_filename}")

        data_tensor = torch.tensor(data_shuffled, dtype=torch.float32)
        n_rows, n_cols = data_tensor.shape

        dataset = CoclusteringDataset(
            data_matrix=data_tensor,
            batch_size=64,
            normalize=True,
            row_labels=row_labels_true,
            col_labels=col_labels_true
        )

        row_input_dim = n_cols
        col_input_dim = n_rows
        row_hidden_dims = [128, 128]
        col_hidden_dims = [128, 128]
        row_latent_dim = 10
        col_latent_dim = 10
        row_components = 20
        col_components = 20
        
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
            lambda_mi=1.0,
            device=device
        )

        print(f"\n[Noise={int(noise_frac*100)}%] Pre-training row model...")
        pretrain_vae(model.row_model, dataset.row_loader, device, n_pretrain_epochs=5, learning_rate=1e-3)
        initialize_gmm_parameters(model.row_model, dataset.row_loader, device)

        print(f"[Noise={int(noise_frac*100)}%] Pre-training column model...")
        pretrain_vae(model.col_model, dataset.col_loader, device, n_pretrain_epochs=5, learning_rate=1e-3)
        initialize_gmm_parameters(model.col_model, dataset.col_loader, device)

        row_clusters_init, col_clusters_init = model.get_cluster_assignments(
            dataset.row_data.to(device),
            dataset.col_data.to(device)
        )
        init_plot_filename = f"initial_coclusters_{int(noise_frac*100)}percent_noise.png"
        plot_rearranged_coclusters(
            data_matrix=data_tensor.cpu().numpy(),
            row_clusters=row_clusters_init.cpu().numpy(),
            col_clusters=col_clusters_init.cpu().numpy(),
            title=f"Initial Rearranged Checkerboard ({int(noise_frac*100)}% Noise)",
            filename=init_plot_filename
        )
        print(f"Saved: {init_plot_filename}")

        beta_schedule = [min(1.0, 0.01 + i * 0.1) for i in range(5)] + [1.0]*25
        
        trainer = ModelTrainer(
            model=model,
            dataset=dataset,
            learning_rate=1e-3,
            beta_schedule=beta_schedule,
            recon_weight=1.0,
            kl_weight=1.0,
            mi_weight=5.0
        )

        print(f"[Noise={int(noise_frac*100)}%] Training co-clustering...")
        training_results = trainer.train(
            n_epochs=30,
            evaluate_every=5,
            early_stopping_patience=5
        )

        row_clusters_final, col_clusters_final = model.get_cluster_assignments(
            dataset.row_data.to(device),
            dataset.col_data.to(device)
        )
        final_plot_filename = f"final_coclusters_{int(noise_frac*100)}percent_noise.png"
        plot_rearranged_coclusters(
            data_matrix=data_tensor.cpu().numpy(),
            row_clusters=row_clusters_final.cpu().numpy(),
            col_clusters=col_clusters_final.cpu().numpy(),
            title=f"Final Rearranged Checkerboard ({int(noise_frac*100)}% Noise)",
            filename=final_plot_filename
        )
        print(f"Saved: {final_plot_filename}")

        final_eval = trainer.evaluate()
        print(f"[Noise={int(noise_frac*100)}%] Final MI Loss: {final_eval['mutual_info_loss']:.4f}")
        print(f"[Noise={int(noise_frac*100)}%] Final Row Accuracy: {final_eval['row_accuracy']}")
        print(f"[Noise={int(noise_frac*100)}%] Final Row NMI: {final_eval['row_nmi']}")
        print(f"[Noise={int(noise_frac*100)}%] Final Column Accuracy: {final_eval['col_accuracy']}")
        print(f"[Noise={int(noise_frac*100)}%] Final Column NMI: {final_eval['col_nmi']}")

        final_results.append({
            'noise_percent': noise_frac * 100,
            'final_mi_loss': final_eval['mutual_info_loss'],
            'row_accuracy': final_eval['row_accuracy'],
            'row_nmi': final_eval['row_nmi'],
            'col_accuracy': final_eval['col_accuracy'],
            'col_nmi': final_eval['col_nmi']
        })
    
    print("\n==== Summary of Results by Noise Level ====")
    for res in final_results:
        print(
            f"Noise={res['noise_percent']}% | "
            f"MI Loss={res['final_mi_loss']:.4f}, "
            f"Row Acc={res['row_accuracy']}, Row NMI={res['row_nmi']}, "
            f"Col Acc={res['col_accuracy']}, Col NMI={res['col_nmi']}"
        )


if __name__ == '__main__':
    main()
