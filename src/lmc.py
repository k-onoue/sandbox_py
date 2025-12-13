"""
Linear Model of Coregionalization (LMC) Gaussian Process with Variable Rank Support.

This module implements an LMC-GP model that supports:
- Inhomogeneous training data (different observation points per task)
- Variable rank coregionalization matrices per latent function
- Custom SimplexSGD optimizer for constrained optimization on the simplex
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =============================================================================
# Optimizer
# =============================================================================

class SimplexSGD(optim.Optimizer):
    """
    SGD optimizer using explicit Euclidean gradients on the simplex.
    
    Optimizes parameters that represent probability distributions via softmax,
    using the natural gradient in expectation parameter space.
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 0.1, 
        line_search: bool = False, 
        ls_max_iter: int = 10, 
        ls_beta: float = 0.5, 
        ls_c1: float = 1e-4
    ):
        defaults = dict(
            lr=lr, 
            line_search=line_search, 
            ls_max_iter=ls_max_iter,
            ls_beta=ls_beta, 
            ls_c1=ls_c1
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            line_search = group['line_search']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Expectation parameters via softmax (sum to 1 along dim=0)
                w = torch.softmax(p, dim=0)
                
                # Convert gradient: dL/dw = dL/dtheta / w
                eps = 1e-12
                d_w = p.grad / (w + eps)
                
                # Center the gradient (remove drift in theta space)
                d_w = d_w - d_w.mean(dim=0, keepdim=True)
                
                # Search direction
                search_dir = -d_w
                
                # Determine step size
                if line_search and closure is not None:
                    alpha = self._line_search(p, search_dir, closure, group, loss)
                else:
                    alpha = lr
                
                p.add_(alpha * search_dir)
                
        return loss
    
    def _line_search(self, param, direction, closure, group, current_loss) -> float:
        """Backtracking line search with Armijo condition."""
        alpha = group['lr']
        beta = group['ls_beta']
        c1 = group['ls_c1']
        max_iter = group['ls_max_iter']
        
        param_orig = param.clone()
        grad_dot_dir = (param.grad * direction).sum()
        
        if grad_dot_dir >= 0:
            return alpha * 0.1
        
        for _ in range(max_iter):
            param.copy_(param_orig + alpha * direction)
            with torch.enable_grad():
                new_loss = closure()
            
            if new_loss.item() <= current_loss.item() + c1 * alpha * grad_dot_dir.item():
                param.copy_(param_orig)
                return alpha
            
            alpha *= beta
        
        param.copy_(param_orig)
        return alpha


# =============================================================================
# LMC Gaussian Process Model
# =============================================================================

class LMCGaussianProcess(nn.Module):
    """
    Linear Model of Coregionalization (LMC) GP with variable rank support.
    
    Supports inhomogeneous data where each task can have different observation points.
    
    Input format:
        X: (N_total, input_dim + 1) - last column is task indicator (0, 1, ..., D-1)
        y: (N_total,) - flattened observations
    
    Args:
        num_outputs: Number of output tasks (D)
        num_latents: Number of latent functions (Q)
        ranks: List of ranks for each latent's coregionalization matrix.
               If None, defaults to rank 1 for all latents.
        input_dim: Dimension of input space (excluding task indicator)
    """
    
    def __init__(
        self, 
        num_outputs: int, 
        num_latents: int, 
        ranks: Optional[List[int]] = None, 
        input_dim: int = 1
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_latents = num_latents
        self.input_dim = input_dim
        
        # Handle ranks configuration
        if ranks is None:
            self.ranks = [1] * num_latents
        else:
            assert len(ranks) == num_latents, "ranks list must match num_latents"
            self.ranks = ranks
        
        # Kernel hyperparameters
        self.log_lengthscales = nn.Parameter(torch.zeros(num_latents))
        self.log_scales = nn.Parameter(torch.zeros(num_outputs))
        self.log_noise_sigma = nn.Parameter(torch.tensor(-2.0))

        # Mixing weights (variable rank per latent)
        self.logits_list = nn.ParameterList([
            nn.Parameter(torch.randn(num_outputs, self.ranks[q]) * 0.1)
            for q in range(num_latents)
        ])

    # --- Properties ---
    
    @property
    def lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscales)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def noise_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_noise_sigma)

    # --- Mixing weights access ---
    
    def get_mixing_matrix(self, q: int) -> torch.Tensor:
        """Get normalized mixing matrix W_q for latent q. Shape: (D, rank_q)"""
        return torch.softmax(self.logits_list[q], dim=0)

    def get_coregionalization_matrix(self, q: int) -> torch.Tensor:
        """Compute B_q = A_q @ A_q.T for latent q. Shape: (D, D)"""
        S = torch.diag(self.scales)
        W_q = self.get_mixing_matrix(q)
        A_q = S @ W_q
        return A_q @ A_q.T

    def get_total_coregionalization_matrix(self) -> torch.Tensor:
        """Compute total coregionalization B = sum_q B_q. Shape: (D, D)"""
        return sum(self.get_coregionalization_matrix(q) for q in range(self.num_latents))

    # --- Kernel computation ---
    
    def _extract_x_and_task(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input into spatial coordinates and task indices."""
        return X[:, :-1], X[:, -1].long()

    def compute_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix for inhomogeneous data.
        
        Args:
            X1: (N1, input_dim + 1) with task indicators
            X2: (N2, input_dim + 1) with task indicators
            
        Returns:
            K: (N1, N2) kernel matrix
        """
        x1, task1 = self._extract_x_and_task(X1)
        x2, task2 = self._extract_x_and_task(X2)
        
        s = self.scales.view(-1, 1)
        ls = self.lengthscales
        
        K = torch.zeros(X1.shape[0], X2.shape[0])
        
        for q in range(self.num_latents):
            # Spatial kernel (RBF)
            dist_sq = torch.cdist(x1, x2, p=2) ** 2
            K_spatial = torch.exp(-0.5 * dist_sq / (ls[q] ** 2))
            
            # Task correlation via mixing matrix
            W_q = self.get_mixing_matrix(q)
            A_q = s * W_q  # (D, rank_q)
            
            # Extract rows for observed tasks
            A_obs1 = A_q[task1]  # (N1, rank_q)
            A_obs2 = A_q[task2]  # (N2, rank_q)
            
            # Coregionalization contribution
            B_block = A_obs1 @ A_obs2.T  # (N1, N2)
            
            K += B_block * K_spatial
        
        return K

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute full covariance matrix K(X, X) + noise."""
        K = self.compute_kernel(X, X)
        return K + self.noise_sigma ** 2 * torch.eye(K.shape[0])

    def negative_log_likelihood(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute negative log marginal likelihood."""
        K = self.forward(X)
        
        # Cholesky decomposition with jitter
        L = torch.linalg.cholesky(K + 1e-5 * torch.eye(K.shape[0]))
        alpha = torch.cholesky_solve(y.unsqueeze(1), L)
        
        data_fit = 0.5 * y @ alpha.squeeze()
        complexity = torch.sum(torch.log(torch.diag(L)))
        const = 0.5 * len(y) * math.log(2 * math.pi)
        
        return data_fit + complexity + const

    # --- Prediction ---
    
    def predict(
        self, 
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        x_test: torch.Tensor, 
        task_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior predictive distribution for a specific task.
        
        Args:
            X_train: Training inputs with task indicators
            y_train: Training observations
            x_test: Test inputs (without task indicator)
            task_idx: Which task to predict
            
        Returns:
            mean: Predictive mean
            std: Predictive standard deviation
        """
        self.eval()
        with torch.no_grad():
            n_test = x_test.shape[0]
            
            # Create test inputs with task indicator
            task_indicator = torch.full((n_test, 1), task_idx, dtype=x_test.dtype)
            X_test = torch.cat([x_test, task_indicator], dim=1)
            
            # Training covariance
            K_ff = self.forward(X_train)
            L = torch.linalg.cholesky(K_ff + 1e-5 * torch.eye(K_ff.shape[0]))
            alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
            
            # Cross-covariance and test covariance
            K_sf = self.compute_kernel(X_test, X_train)
            K_ss_diag = torch.diagonal(self.compute_kernel(X_test, X_test))
            
            # Posterior mean
            mean = (K_sf @ alpha).squeeze()
            
            # Posterior variance
            v = torch.linalg.solve_triangular(L, K_sf.T, upper=False)
            var = K_ss_diag - torch.sum(v ** 2, dim=0) + self.noise_sigma ** 2
            std = torch.sqrt(torch.clamp(var, min=1e-10))
            
            return mean, std


# =============================================================================
# Data Utilities
# =============================================================================

def prepare_inhomogeneous_data(
    x_list: List[torch.Tensor], 
    y_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare inhomogeneous multi-task data.
    
    Args:
        x_list: List of input tensors per task, x_list[j] has shape (N_j, input_dim)
        y_list: List of output tensors per task, y_list[j] has shape (N_j,)
        
    Returns:
        X: (N_total, input_dim + 1) with task indicators
        y: (N_total,) flattened observations
    """
    X_parts, y_parts = [], []
    
    for j, (x_j, y_j) in enumerate(zip(x_list, y_list)):
        task_indicator = torch.full((x_j.shape[0], 1), j, dtype=x_j.dtype)
        X_parts.append(torch.cat([x_j, task_indicator], dim=1))
        y_parts.append(y_j.flatten())
    
    return torch.cat(X_parts, dim=0), torch.cat(y_parts, dim=0)


# =============================================================================
# Training
# =============================================================================

def train_lmc(
    model: LMCGaussianProcess, 
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    num_epochs: int = 200,
    lr_hyper: float = 0.01,
    lr_simplex: float = 1.0,
    print_every: int = 20
) -> List[float]:
    """
    Train the LMC model using alternating optimization.
    
    Args:
        model: LMCGaussianProcess instance
        X_train: Training inputs with task indicators
        y_train: Training observations
        num_epochs: Number of training epochs
        lr_hyper: Learning rate for hyperparameters (Adam)
        lr_simplex: Learning rate for simplex parameters
        print_every: Print frequency
        
    Returns:
        loss_history: List of NLL values per epoch
    """
    # Split parameters
    simplex_params = list(model.logits_list.parameters())
    hyper_params = [model.log_lengthscales, model.log_scales, model.log_noise_sigma]
    
    # Optimizers
    optimizer_hyper = optim.Adam(hyper_params, lr=lr_hyper)
    optimizer_simplex = SimplexSGD(simplex_params, lr=lr_simplex, line_search=True)
    
    loss_history = []
    
    print("Training LMC model...")
    for epoch in range(num_epochs):
        model.train()
        
        # Step 1: Update hyperparameters
        optimizer_hyper.zero_grad()
        loss = model.negative_log_likelihood(X_train, y_train)
        loss.backward()
        optimizer_hyper.step()
        
        # Step 2: Update simplex weights
        def closure():
            optimizer_simplex.zero_grad()
            loss_val = model.negative_log_likelihood(X_train, y_train)
            loss_val.backward()
            return loss_val

        optimizer_simplex.zero_grad()
        model.negative_log_likelihood(X_train, y_train).backward()
        optimizer_simplex.step(closure)
        
        loss_history.append(loss.item())
        
        if epoch % print_every == 0:
            print(f"  Epoch {epoch:03d} | NLL: {loss.item():.4f}")

    print("Training complete.\n")
    return loss_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_predictions(
    model: LMCGaussianProcess,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    x_range: Tuple[float, float] = (-2, 13),
    n_test: int = 200
) -> None:
    """Plot GP predictions for each task."""
    model.eval()
    num_tasks = model.num_outputs
    
    x_test = torch.linspace(x_range[0], x_range[1], n_test).unsqueeze(1)
    
    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 3 * num_tasks), sharex=True)
    if num_tasks == 1:
        axes = [axes]
    
    with torch.no_grad():
        for j in range(num_tasks):
            mean, std = model.predict(X_train, y_train, x_test, task_idx=j)
            
            # Extract training data for this task
            mask = X_train[:, -1] == j
            train_x = X_train[mask, 0]
            train_y = y_train[mask]
            
            ax = axes[j]
            ax.scatter(train_x, train_y, c='black', marker='x', label='Observations', zorder=5)
            ax.plot(x_test.numpy(), mean.numpy(), 'b-', label='Predictive Mean')
            ax.fill_between(
                x_test.squeeze().numpy(),
                (mean - 2 * std).numpy(),
                (mean + 2 * std).numpy(),
                alpha=0.3, color='blue', label='95% Confidence'
            )
            ax.set_title(f'Task {j}')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc='upper right')

    axes[-1].set_xlabel('x')
    plt.tight_layout()
    plt.show()


def visualize_coregionalization(model: LMCGaussianProcess) -> None:
    """Visualize learned coregionalization matrices."""
    model.eval()
    num_latents = model.num_latents
    
    fig, axes = plt.subplots(1, num_latents + 1, figsize=(4 * (num_latents + 1), 3.5))
    
    with torch.no_grad():
        matrices = [model.get_coregionalization_matrix(q) for q in range(num_latents)]
        B_total = model.get_total_coregionalization_matrix()
        matrices.append(B_total)
        
        # Unified colormap range
        vmin = min(m.min().item() for m in matrices)
        vmax = max(m.max().item() for m in matrices)
        
        # Plot individual B_q
        for q in range(num_latents):
            ax = axes[q]
            B_q = matrices[q]
            im = ax.imshow(B_q, cmap='coolwarm', vmin=vmin, vmax=vmax)
            
            num_rank = torch.linalg.matrix_rank(B_q, tol=1e-3).item()
            ax.set_title(f"Latent {q}\nRank: {model.ranks[q]} (numeric: {num_rank})")
            ax.set_xlabel("Task")
            if q == 0:
                ax.set_ylabel("Task")

        # Plot total B
        ax = axes[-1]
        im = ax.imshow(B_total, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title("Total B\n(Sum of B_q)")
        ax.set_xlabel("Task")
        
        # Colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Covariance')
        
    plt.suptitle("Learned Task Correlations", fontsize=14)
    plt.show()


def visualize_results(
    model: LMCGaussianProcess,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    x_range: Tuple[float, float] = (-2, 13)
) -> None:
    """Unified visualization of predictions and coregionalization."""
    print("Generating prediction plots...")
    visualize_predictions(model, X_train, y_train, x_range)
    
    print("Generating coregionalization plots...")
    visualize_coregionalization(model)


# =============================================================================
# Main
# =============================================================================

def create_synthetic_data() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate synthetic inhomogeneous multi-task data."""
    # Different sample sizes and ranges per task
    N1, N2, N3 = 40, 50, 30
    
    x1 = torch.linspace(0, 10, N1).unsqueeze(1)
    x2 = torch.linspace(1, 9, N2).unsqueeze(1)
    x3 = torch.linspace(2, 12, N3).unsqueeze(1)
    
    # Latent functions
    u1 = lambda x: torch.sin(x * 0.5)
    u2 = lambda x: torch.cos(x * 1.5)
    
    # Task outputs (different mixtures of latent functions)
    y1 = (1.0 * u1(x1) + 0.1 * u2(x1) + torch.randn(N1, 1) * 0.1).flatten()
    y2 = (0.1 * u1(x2) + 1.0 * u2(x2) + torch.randn(N2, 1) * 0.1).flatten()
    y3 = (0.5 * u1(x3) + 0.5 * u2(x3) + torch.randn(N3, 1) * 0.1).flatten()
    
    return [x1, x2, x3], [y1, y2, y3]


def main():
    torch.manual_seed(42)
    
    # Create data
    x_list, y_list = create_synthetic_data()
    X_train, y_train = prepare_inhomogeneous_data(x_list, y_list)
    
    print(f"Data: {X_train.shape[0]} total points across {len(x_list)} tasks")
    print(f"Points per task: {[x.shape[0] for x in x_list]}\n")
    
    # Model configuration
    num_outputs = 3
    num_latents = 3
    ranks = [3, 3, 3]
    
    model = LMCGaussianProcess(
        num_outputs=num_outputs,
        num_latents=num_latents,
        ranks=ranks
    )
    print(f"Model: {num_latents} latents with ranks {ranks}\n")
    
    # Train
    train_lmc(model, X_train, y_train, num_epochs=300)
    
    # Print learned parameters
    with torch.no_grad():
        print("Learned Parameters:")
        print(f"  Lengthscales: {model.lengthscales.numpy()}")
        print(f"  Scales: {model.scales.numpy()}")
        print(f"  Noise sigma: {model.noise_sigma.item():.4f}\n")
        
        for q in range(num_latents):
            B_q = model.get_coregionalization_matrix(q)
            num_rank = torch.linalg.matrix_rank(B_q, tol=1e-4).item()
            print(f"  Latent {q}: config rank={ranks[q]}, numeric rank={num_rank}")
    
    # Visualize
    print()
    visualize_results(model, X_train, y_train)


if __name__ == "__main__":
    main()