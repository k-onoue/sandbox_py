import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Optimizer: Our Method (Explicit Euclidean Gradient on Simplex) ---
class SimplexSGD(optim.Optimizer):
    """
    期待値パラメータのユークリッド勾配を明示的に利用するオプティマイザ。
    (論文のCauchy-Simplexと考え方は近いが、Line Searchを追加して最適なステップサイズを自動決定)
    """
    def __init__(self, params, lr=0.1, line_search=False, ls_max_iter=10, ls_beta=0.5, ls_c1=1e-4):
        """
        Args:
            params: Parameters to optimize
            lr: Initial learning rate (or max step size if line_search=True)
            line_search: Enable Armijo backtracking line search
            ls_max_iter: Maximum line search iterations
            ls_beta: Backtracking factor (0 < beta < 1)
            ls_c1: Armijo condition parameter (sufficient decrease)
        """
        defaults = dict(lr=lr, line_search=line_search, ls_max_iter=ls_max_iter, 
                       ls_beta=ls_beta, ls_c1=ls_c1)
        super(SimplexSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (required for line search)
        """
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
                
                # w (Expectation Param)
                w = torch.softmax(p, dim=0)
                
                # dL/dtheta
                d_theta = p.grad
                
                # dL/dw = dL/dtheta / w (Explicit Euclidean Gradient)
                eps = 1e-12
                d_w = d_theta / (w + eps)
                
                # Center the gradient (remove drift in theta space)
                d_w = d_w - d_w.mean()
                
                # Store gradient direction for line search
                search_dir = -d_w
                
                # Line search
                if line_search and closure is not None:
                    alpha = self._line_search(p, search_dir, closure, group, loss)
                else:
                    alpha = lr
                
                # Update: Move in the direction of negative natural gradient
                p.add_(alpha * search_dir)
                
        return loss
    
    def _line_search(self, param, direction, closure, group, current_loss):
        """
        Armijo backtracking line search
        
        Args:
            param: Parameter being optimized
            direction: Search direction
            closure: Loss function closure
            group: Parameter group with hyperparameters
            current_loss: Current loss value
            
        Returns:
            Step size alpha
        """
        alpha = group['lr']
        beta = group['ls_beta']
        c1 = group['ls_c1']
        max_iter = group['ls_max_iter']
        
        # Save original parameter value
        param_orig = param.clone()
        
        # Compute directional derivative (should be negative for descent)
        # grad^T * direction
        grad_dot_dir = (param.grad * direction).sum()
        
        # If direction is not a descent direction, return small step
        if grad_dot_dir >= 0:
            return alpha * 0.1
        
        # Armijo backtracking
        for i in range(max_iter):
            # Try step: param = param_orig + alpha * direction
            param.copy_(param_orig + alpha * direction)
            
            # Evaluate loss at new point
            with torch.enable_grad():
                new_loss = closure()
            
            # Check Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
            if new_loss.item() <= current_loss.item() + c1 * alpha * grad_dot_dir.item():
                # Sufficient decrease achieved
                param.copy_(param_orig)  # Restore for proper update in step()
                return alpha
            
            # Reduce step size
            alpha *= beta
        
        # If line search fails, restore and return small step
        param.copy_(param_orig)
        return alpha

# --- 2. Data Generation (According to Paper Sec 6.2) ---
def generate_exam_data(n_students=200, n_questions=75):
    # Difficulty q_i [cite: 413]
    q = torch.zeros(n_questions)
    q[:60] = 7/8  # Easy
    q[60:] = 1/5  # Hard
    
    # Smartness s_j [cite: 413]
    s = torch.zeros(n_students)
    s[:120] = 7/10 # Smart
    s[120:] = 1/2  # Average
    
    # Probability Matrix P_ij = q_i * s_j
    # Bernoulli sampling [cite: 412]
    probs = torch.outer(q, s)
    X = torch.bernoulli(probs) # [n_questions, n_students]
    
    return X

# --- 3. Model & Loss (KDE & KL Divergence) ---
class ExamWeightingModel(nn.Module):
    def __init__(self, n_questions):
        super().__init__()
        # Initial weights: Uniform (1/n) [cite: 479]
        # We store as theta (logits) initialized to zeros
        self.theta = nn.Parameter(torch.zeros(n_questions))
        
    @property
    def weights(self):
        return torch.softmax(self.theta, dim=0)

def gaussian_kde(samples, grid, bandwidth=0.05):
    """
    Differentiable Kernel Density Estimation
    rho_epsilon(z) [cite: 392, 394]
    """
    # samples: [n_students]
    # grid: [n_grid_points]
    n = samples.shape[0]
    
    # (x - X_j) / epsilon
    diff = (grid.unsqueeze(1) - samples.unsqueeze(0)) / bandwidth
    
    # Standard Normal PDF
    pdf = torch.exp(-0.5 * diff**2) / np.sqrt(2 * np.pi)
    
    # Average over samples and divide by bandwidth
    density = pdf.sum(dim=1) / (n * bandwidth)
    return density

def target_distribution(grid):
    """
    Target f: Truncated Normal (mean=0.5, std=0.1) [cite: 416]
    """
    mu = 0.5
    sigma = 0.1
    pdf = torch.exp(-0.5 * ((grid - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    # Simple truncation/normalization over the grid [0, 1]
    pdf = pdf / pdf.sum() # Normalize to sum to 1 on the grid for KL calculation
    return pdf

def riemann_kl_divergence(p, q, dx):
    """
    Riemann approximation of KL Divergence [cite: 400]
    sum p * log(p/q) * dx
    """
    # Numerical stability
    p = p + 1e-12
    q = q + 1e-12
    
    # Normalize densities to valid PMFs on the grid for correct KL calculation
    p = p / (p.sum() * dx)
    q = q / (q.sum() * dx)
    
    return (p * torch.log(p / q)).sum() * dx

# --- 4. Main Experiment ---
def run_comparison(use_line_search=False):
    # Setup
    n_students = 200
    n_questions = 75
    X = generate_exam_data(n_students, n_questions)
    
    model = ExamWeightingModel(n_questions)
    
    # Optimizer: Our Method with optional line search
    optimizer = SimplexSGD(
        model.parameters(), 
        lr=10.0,  # Max step size for line search, or fixed LR
        line_search=use_line_search,
        ls_max_iter=10,
        ls_beta=0.5,
        ls_c1=1e-4
    )
    
    # Grid for Riemann Sum [0, 1] with 400 steps [cite: 417]
    grid = torch.linspace(0, 1, 401)
    dx = 1.0 / 400
    target_pdf = target_distribution(grid)
    
    loss_history = []
    
    # Define closure for line search
    def closure():
        optimizer.zero_grad()
        w = model.weights
        student_scores = torch.matmul(w, X)
        estimated_pdf = gaussian_kde(student_scores, grid, bandwidth=0.05)
        loss = riemann_kl_divergence(estimated_pdf, target_pdf, dx)
        loss.backward()
        return loss
    
    mode_str = "with Line Search" if use_line_search else "Fixed LR"
    print(f"\n=== SimplexSGD ({mode_str}) ===")
    print("Iter | KL Divergence | Max Weight")
    print("-" * 35)
    
    # Loop for 150 iterations (Same as Paper [cite: 418])
    for i in range(151):
        if use_line_search:
            # Line search requires closure
            loss = closure()
            optimizer.step(closure)
        else:
            # Fixed learning rate mode
            optimizer.zero_grad()
            w = model.weights
            student_scores = torch.matmul(w, X)
            estimated_pdf = gaussian_kde(student_scores, grid, bandwidth=0.05)
            loss = riemann_kl_divergence(estimated_pdf, target_pdf, dx)
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        
        if i % 10 == 0 or i == 150:
            w = model.weights
            print(f"{i:4d} | {loss.item():.6f}    | {w.max().item():.4f}")

    # --- Analysis & Visualization ---
    
    # Paper Table 2 Comparison:
    # Best methods (PFW) achieved ~0.009 - 0.030 range.
    final_loss = loss_history[-1]
    print("-" * 35)
    print(f"Final KL Divergence: {final_loss:.6f}")
    print(f"Paper's Typical Range: 0.009 - 0.035 ")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Curve
    ax1.plot(loss_history, label=f'SimplexSGD ({mode_str})', linewidth=2)
    ax1.set_title('Convergence of KL Divergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('KL Divergence')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Final Distributions
    with torch.no_grad():
        final_scores = torch.matmul(model.weights, X)
        final_pdf = gaussian_kde(final_scores, grid, bandwidth=0.05)
        
        # Normalize for plotting
        target_plot = target_pdf / (target_pdf.sum() * dx)
        final_plot = final_pdf / (final_pdf.sum() * dx)

    ax2.plot(grid.numpy(), target_plot.numpy(), label='Target (Normal)', linestyle='--', linewidth=2, color='black')
    ax2.plot(grid.numpy(), final_plot.numpy(), label='Estimated (Our Method)', linewidth=2, color='tab:blue')
    ax2.hist(final_scores.numpy(), bins=30, density=True, alpha=0.3, color='tab:blue', label='Weighted Scores Hist')
    ax2.set_title(f'Distributions at Iter 150 (KL={final_loss:.4f})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return loss_history

if __name__ == "__main__":
    # Run with line search enabled
    run_comparison(use_line_search=True)
    
    # Uncomment to compare with fixed learning rate:
    # run_comparison(use_line_search=False)