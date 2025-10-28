import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import json
import argparse 
import warnings

# Suppress UserWarning from matplotlib about log scale, etc.
warnings.filterwarnings("ignore", category=UserWarning)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ==============================================================================
# The core functions (beta_t, ScoreNet, etc.) remain the same
# For brevity, they are included here without change.
# ==============================================================================

def beta_t(t, beta_0, beta_T):
    return beta_0 + t * (beta_T - beta_0)

def alpha_bar_t(t, beta_0, beta_T):
    integral_beta = 0.5 * t * (2 * beta_0 + t * (beta_T - beta_0))
    return torch.exp(-0.5 * integral_beta)

def get_diffusion_params(t, beta_0, beta_T, epsilon_t, device):
    t_clipped = torch.clamp(t, min=epsilon_t).to(device)
    _alpha_bar = alpha_bar_t(t_clipped, beta_0, beta_T)
    return _alpha_bar.view(-1, 1), (1.0 - _alpha_bar).sqrt().view(-1, 1)

class ScoreNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x, t):
        x_with_time = torch.cat([x, t.view(-1, 1)], dim=1)
        return self.net(x_with_time)

def compute_hovr_regularization(model, x, t, k):
    if k == 0 or k is None:
        return torch.tensor(0.0, device=x.device)
    t_detached = t.detach()
    x_var = x.clone().detach().requires_grad_(True)
    y = model(x_var, t_detached)
    for i in range(k):
        grad_outputs = torch.ones_like(y)
        grads = torch.autograd.grad(outputs=y, inputs=x_var, grad_outputs=grad_outputs, create_graph=True)[0]
        y = grads
    hovr_loss = (grads**2).sum() / x.shape[0]
    return hovr_loss

@torch.no_grad()
def generate_and_save_samples(model, epoch, true_data, params, exp_dir):
    model.eval()
    device = next(model.parameters()).device
    xt = torch.randn(params["n_samples_viz"], 2, device=device)
    ts = np.linspace(params["T_end"], params["epsilon_t"], params["n_steps_viz"])
    for i in range(params["n_steps_viz"] - 1):
        t_curr_val = ts[i]
        dt = ts[i] - ts[i+1]
        t_curr = torch.full((params["n_samples_viz"],), t_curr_val, device=device)
        f_drift = -0.5 * beta_t(t_curr, params["beta_0"], params["beta_T"]).view(-1, 1) * xt
        g2_term = beta_t(t_curr, params["beta_0"], params["beta_T"]).view(-1, 1)
        _, std_coeff = get_diffusion_params(t_curr, params["beta_0"], params["beta_T"], params["epsilon_t"], device)
        pred_noise = model(xt, t_curr)
        score = -pred_noise / std_coeff
        drift = f_drift - 0.5 * g2_term * score
        xt = xt - drift * dt
    samples = xt.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(true_data[:, 0], true_data[:, 1], s=5, alpha=0.1, color='gray', label='True Data')
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color='blue', label='Generated Samples')
    plt.title(f"Generated Samples at Epoch {epoch}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.axis('equal')
    plot_range = params.get("plot_range", 3.0)
    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, f"samples_epoch_{epoch:05d}.png"))
    plt.close()
    model.train()

def get_dataset(params):
    if params["dataset"] == "two_moons":
        X, _ = make_moons(n_samples=params["n_samples"], noise=0.05)
        X = X * 1.5
        params["plot_range"] = 3.0
    elif params["dataset"] == "swiss_roll":
        X, _ = make_swiss_roll(n_samples=params["n_samples"], noise=0.8)
        X = X[:, [0, 2]] / 7.0
        params["plot_range"] = 2.0
    elif params["dataset"] == "eight_gaussians":
        scale = 2.0
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1./np.sqrt(2), 1./np.sqrt(2)), (1./np.sqrt(2), -1./np.sqrt(2)),
                   (-1./np.sqrt(2), 1./np.sqrt(2)), (-1./np.sqrt(2), -1./np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        X = []
        for i in range(params["n_samples"]):
            point = np.random.randn(2) * 0.1
            center = centers[np.random.randint(len(centers))]
            point[0] += center[0]
            point[1] += center[1]
            X.append(point)
        X = np.array(X)
        params["plot_range"] = 4.0
    else:
        raise ValueError(f"Unknown dataset: {params['dataset']}")
    return X

def run_experiment(lambda_hovr, k_hovr, params, device):
    exp_name = f"dataset={params['dataset']}_lambda={lambda_hovr}_k={k_hovr}"
    exp_dir = os.path.join("hovr_experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f"Starting Experiment: {exp_name}")
    print("="*80)
    X = get_dataset(params)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.3, color='gray')
    plt.title(f"True Data Distribution ({params['dataset']})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('equal')
    plot_range = params.get("plot_range", 3.0)
    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "true_data_distribution.png"))
    plt.close()
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    model = ScoreNet(in_dim=X.shape[1], out_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    losses = []
    pbar = tqdm(range(params["n_epochs"]), desc=f"Training {exp_name}")
    for epoch in pbar:
        epoch_loss = 0.0
        for data, in loader:
            x0 = data.to(device)
            optimizer.zero_grad()
            t = torch.rand(x0.shape[0], device=device) * (params["T_end"] - params["epsilon_t"]) + params["epsilon_t"]
            mean_coeff, std_coeff = get_diffusion_params(t, params["beta_0"], params["beta_T"], params["epsilon_t"], device)
            noise = torch.randn_like(x0)
            xt = mean_coeff * x0 + std_coeff * noise
            predicted_noise = model(xt, t)
            dsm_loss = ((predicted_noise - noise)**2).mean()
            hovr_loss = compute_hovr_regularization(model, xt, t, k_hovr)
            total_loss = dsm_loss + lambda_hovr * hovr_loss
            total_loss.backward()
            if params["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip_norm"])
            optimizer.step()
            epoch_loss += total_loss.item()
        avg_epoch_loss = epoch_loss / len(loader)
        losses.append(avg_epoch_loss)
        pbar.set_postfix({"loss": f"{avg_epoch_loss:.4f}"})
        if epoch % params["save_interval"] == 0:
            generate_and_save_samples(model, epoch, X, params, exp_dir)
    generate_and_save_samples(model, params["n_epochs"], X, params, exp_dir)
    torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pth"))
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f"Loss Curve for {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()
    print(f"Finished Experiment: {exp_name}. Results saved in {exp_dir}")

# ==============================================================================
# --- Main Execution Block with Command-Line Argument Parsing ---
# ==============================================================================

if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Run HOVR Diffusion Model Experiment.")
    parser.add_argument(
        '--params_file', 
        type=str, 
        default='params.json', 
        help='Path to the JSON file containing experiment parameters.'
    )
    args = parser.parse_args()

    # --- Load Parameters from JSON ---
    try:
        with open(args.params_file, 'r') as f:
            PARAMS = json.load(f)
        print(f"Successfully loaded parameters from {args.params_file}")
    except FileNotFoundError:
        print(f"Error: Parameter file '{args.params_file}' not found. Please create it.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.params_file}'. Please check the format.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main_exp_dir = "hovr_experiments"
    os.makedirs(main_exp_dir, exist_ok=True)

    # --- Grid Search ---
    grid = list(itertools.product(PARAMS["k_hovr_values"], PARAMS["lambda_hovr_values"]))
    
    for k, lam in grid:
        if k == 2 and lam == 0.0:
            continue
        
        # Use a copy of params for each run
        run_params = PARAMS.copy()
        run_experiment(lambda_hovr=lam, k_hovr=k, params=run_params, device=device)

    print("\nAll experiments finished.")