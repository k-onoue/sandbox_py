"""
This implementation is based on the code shared by Arthur Zwaenepoel on GitHub.
See https://github.com/arzwa/IncBetaDer for details.
His Google Scholar profile: https://scholar.google.com/citations?user=8VSQd34AAAAJ&hl=en

This version has been vectorized to support tensor inputs for batch processing.
This version includes a fix in the backward pass return statement for gradcheck compatibility.
"""


import torch
from torch.autograd import Function

EPSILON = 1e-12
MIN_APPROX = 3
MAX_APPROX = 200


def _beta(p, q):
    # Uses lgamma for compatibility with older PyTorch versions
    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    return torch.exp(log_beta)


def _Kfun(p, q, x_calc):
    return (x_calc.pow(p) * (1.0 - x_calc).pow(q - 1)) / (p * _beta(p, q))


def _ffun(p, q, x_calc):
    return q * x_calc / (p * (1.0 - x_calc))


def _a1fun(p, q, f):
    return p * f * (q - 1.0) / (q * (p + 1.0))


def _anfun(p, q, f, n):
    if n == 1:
        return _a1fun(p, q, f)
    # This function is already vectorized due to using torch operations
    return (
        p.pow(2)
        * f.pow(2)
        * (n - 1.0)
        * (p + q + n - 2.0)
        * (p + n - 1.0)
        * (q - n)
        / (q.pow(2) * (p + 2 * n - 3.0) * (p + 2 * n - 2.0).pow(2) * (p + 2 * n - 1.0))
    )


def _bnfun(p, q, f, n):
    # This function is already vectorized
    x_ = (
        2 * (p * f + 2 * q) * n**2
        + 2 * (p * f + 2 * q) * (p - 1.0) * n
        + p * q * (p - 2.0 - p * f)
    )
    y_ = q * (p + 2 * n - 2.0) * (p + 2 * n)
    return x_ / y_


def _dK_dp(x, p, q, K, psi_pq, psi_p):
    return K * (torch.log(x) - 1.0 / p + psi_pq - psi_p)


def _dK_dq(x, p, q, K, psi_pq, psi_q):
    return K * (torch.log(1.0 - x) + psi_pq - psi_q)


def _da1_dp(p, q, f):
    return -p * f * (q - 1.0) / (q * (p + 1.0).pow(2))


def _dan_dp(p, q, f, n):
    if n == 1:
        return _da1_dp(p, q, f)
    x = -(n - 1.0) * f.pow(2) * p.pow(2) * (q - n)
    y = (
        (-1.0 + p + q) * 8 * n**3
        + (16 * p.pow(2) + (-44.0 + 20 * q) * p + 26.0 - 24 * q) * n**2
        + (
            10 * p.pow(3)
            + (14 * q - 46.0) * p.pow(2)
            + (-40 * q + 66.0) * p
            - 28.0
            + 24 * q
        )
        * n
        + 2 * p.pow(4)
        + (-13 + 3 * q) * p.pow(3)
        + (16 - 14 * q) * p.pow(2)
        + (-29 + 19 * q) * p
        + 10.0
        - 8 * q
    )
    z = (
        q.pow(2)
        * (p + 2 * n - 3.0).pow(2)
        * (p + 2 * n - 2.0).pow(3)
        * (p + 2 * n - 1.0).pow(2)
    )
    return x * y / z


def _da1_dq(p, q, f):
    return f * p / (q * (p + 1.0))


def _dan_dq(p, q, f, n):
    if n == 1:
        return _da1_dq(p, q, f)
    x = p.pow(2) * f.pow(2) * (n - 1.0) * (p + n - 1.0) * (2 * q + p - 2.0)
    y = q.pow(2) * (p + 2 * n - 3.0) * (p + 2 * n - 2.0).pow(2) * (p + 2 * n - 1.0)
    return x / y


def _dbn_dp(p, q, f, n):
    x = (
        (1.0 - p - q) * 4 * n**2
        + (4 * p - 4.0 + 4 * q - 2 * p.pow(2)) * n
        + p.pow(2) * q
    )
    y = q * (p + 2 * n - 2.0).pow(2) * (p + 2 * n).pow(2)
    return p * f * x / y


def _dbn_dq(p, q, f, n):
    return -p.pow(2) * f / (q * (p + 2 * n - 2.0) * (p + 2 * n))


def _dnextapp(an, bn, dan, dbn, Xpp, Xp, dXpp, dXp):
    return dan * Xpp + an * dXpp + dbn * Xp + bn * dXp


class Betainc(Function):
    @staticmethod
    def forward(ctx, a, b, x):
        ctx.save_for_backward(a, b, x)
        
        # --- VECTORIZATION CHANGE: Handle broadcasting ---
        # Get the final shape after broadcasting
        final_shape = torch.broadcast_shapes(a.shape, b.shape, x.shape)
        a = a.expand(final_shape)
        b = b.expand(final_shape)
        x = x.expand(final_shape)

        # --- VECTORIZATION CHANGE: Use masks for edge cases ---
        # Create masks for values outside the (0, 1) range
        x_le_0 = x <= 0.0
        x_ge_1 = x >= 1.0
        x_intermediate = ~x_le_0 & ~x_ge_1

        # The final result tensor, initialized to zeros
        final_I = torch.zeros_like(x)
        final_I[x_ge_1] = 1.0 # Set result to 1 where x >= 1
        
        # Only compute for values strictly between 0 and 1
        if x_intermediate.any():
            a_comp = a[x_intermediate]
            b_comp = b[x_intermediate]
            x_comp = x[x_intermediate]

            swapped = x_comp > a_comp / (a_comp + b_comp)
            
            p = torch.where(swapped, b_comp, a_comp)
            q = torch.where(swapped, a_comp, b_comp)
            x_calc = torch.where(swapped, 1.0 - x_comp, x_comp)

            K = _Kfun(p, q, x_calc)
            f = _ffun(p, q, x_calc)

            # --- VECTORIZATION CHANGE: Initialize state variables as tensors ---
            one = torch.ones_like(p)
            zero = torch.zeros_like(p)
            App, Ap, Bpp, Bp = one.clone(), one.clone(), zero.clone(), one.clone()
            Ixpq = torch.full_like(p, torch.nan)
            
            # --- VECTORIZATION CHANGE: Use a convergence mask in the loop ---
            converged_mask = torch.zeros_like(p, dtype=torch.bool)
            
            for n in range(1, MAX_APPROX + 1):
                if converged_mask.all():
                    break # All elements have converged

                an, bn = _anfun(p, q, f, n), _bnfun(p, q, f, n)
                An, Bn = an * App + bn * Ap, an * Bpp + bn * Bp
                
                # Prevent division by zero, clamp small values
                Bn = torch.where(torch.abs(Bn) < 1e-30, torch.full_like(Bn, 1e-30), Bn)

                Cn = An / Bn
                Ixpqn = K * Cn
                
                # --- VECTORIZATION CHANGE: Update mask and values selectively ---
                not_converged = ~converged_mask
                
                # Identify newly converged elements in this iteration
                newly_converged = (torch.abs(Ixpqn - Ixpq) < EPSILON) & not_converged & (n >= MIN_APPROX)
                converged_mask[newly_converged] = True
                
                # Only update the values for elements that have not yet converged
                Ixpq = torch.where(not_converged, Ixpqn, Ixpq)
                
                # Update state variables for the next iteration
                App, Ap = torch.where(not_converged.unsqueeze(0), torch.stack([Ap, An]), torch.stack([App, Ap]))
                Bpp, Bp = torch.where(not_converged.unsqueeze(0), torch.stack([Bp, Bn]), torch.stack([Bpp, Bp]))

            # Place the computed results back into the final tensor
            # If swapped, result is 1 - Ixpq
            intermediate_result = torch.where(swapped, 1.0 - Ixpq, Ixpq)
            final_I[x_intermediate] = intermediate_result

        return final_I

    @staticmethod
    def backward(ctx, grad_output):
        a, b, x = ctx.saved_tensors
        grad_a = grad_b = grad_x = None

        final_shape = torch.broadcast_shapes(a.shape, b.shape, x.shape)
        a = a.expand(final_shape)
        b = b.expand(final_shape)
        x = x.expand(final_shape)

        x_le_0 = x <= 0.0
        x_ge_1 = x >= 1.0
        x_intermediate = ~x_le_0 & ~x_ge_1

        # Initialize gradients to zero tensors ONLY if they are needed
        if ctx.needs_input_grad[0]: grad_a = torch.zeros_like(a)
        if ctx.needs_input_grad[1]: grad_b = torch.zeros_like(b)
        if ctx.needs_input_grad[2]: grad_x = torch.zeros_like(x)

        if x_intermediate.any():
            a_comp = a[x_intermediate]
            b_comp = b[x_intermediate]
            x_comp = x[x_intermediate]

            swapped = x_comp > a_comp / (a_comp + b_comp)
            
            p = torch.where(swapped, b_comp, a_comp)
            q = torch.where(swapped, a_comp, b_comp)
            x_calc = torch.where(swapped, 1.0 - x_comp, x_comp)

            K = _Kfun(p, q, x_calc)
            f = _ffun(p, q, x_calc)
            psi_p, psi_q, psi_pq = (
                torch.special.digamma(p),
                torch.special.digamma(q),
                torch.special.digamma(p + q),
            )
            dK_d_p = _dK_dp(x_calc, p, q, K, psi_pq, psi_p)
            dK_d_q = _dK_dq(x_calc, p, q, K, psi_pq, psi_q)

            one = torch.ones_like(p)
            zero = torch.zeros_like(p)
            App, Ap, Bpp, Bp = one.clone(), one.clone(), zero.clone(), one.clone()
            dApp_dp, dAp_dp, dBpp_dp, dBp_dp = [z.clone() for z in [zero, zero, zero, zero]]
            dApp_dq, dAp_dq, dBpp_dq, dBp_dq = [z.clone() for z in [zero, zero, zero, zero]]
            
            Ixpq = torch.full_like(p, torch.nan)
            dI_dp = torch.zeros_like(p)
            dI_dq = torch.zeros_like(p)
            
            converged_mask = torch.zeros_like(p, dtype=torch.bool)
            
            for n in range(1, MAX_APPROX + 1):
                if converged_mask.all():
                    break
                    
                an, bn = _anfun(p, q, f, n), _bnfun(p, q, f, n)
                An, Bn = an * App + bn * Ap, an * Bpp + bn * Bp
                
                dan_p, dbn_p = _dan_dp(p, q, f, n), _dbn_dp(p, q, f, n)
                dAn_dp = _dnextapp(an, bn, dan_p, dbn_p, App, Ap, dApp_dp, dAp_dp)
                dBn_dp = _dnextapp(an, bn, dan_p, dbn_p, Bpp, Bp, dBpp_dp, dBp_dp)
                
                dan_q, dbn_q = _dan_dq(p, q, f, n), _dbn_dq(p, q, f, n)
                dAn_dq = _dnextapp(an, bn, dan_q, dbn_q, App, Ap, dApp_dq, dAp_dq)
                dBn_dq = _dnextapp(an, bn, dan_q, dbn_q, Bpp, Bp, dBpp_dq, dBp_dq)

                Bn_safe = torch.where(torch.abs(Bn) < 1e-30, torch.full_like(Bn, 1e-30), Bn)
                
                Cn = An / Bn_safe
                Ixpqn = K * Cn
                
                current_dI_dp = dK_d_p * Cn + K * ((dAn_dp / Bn_safe) - (An * dBn_dp / Bn_safe.pow(2)))
                current_dI_dq = dK_d_q * Cn + K * ((dAn_dq / Bn_safe) - (An * dBn_dq / Bn_safe.pow(2)))
                
                not_converged = ~converged_mask
                newly_converged = (torch.abs(Ixpqn - Ixpq) < EPSILON) & not_converged & (n >= MIN_APPROX)
                converged_mask[newly_converged] = True
                
                Ixpq = torch.where(not_converged, Ixpqn, Ixpq)
                dI_dp = torch.where(not_converged, current_dI_dp, dI_dp)
                dI_dq = torch.where(not_converged, current_dI_dq, dI_dq)
                
                App_s, Ap_s = torch.stack([App, Ap])
                Bpp_s, Bp_s = torch.stack([Bpp, Bp])
                dApp_dp_s, dAp_dp_s = torch.stack([dApp_dp, dAp_dp])
                dBpp_dp_s, dBp_dp_s = torch.stack([dBpp_dp, dBp_dp])
                dApp_dq_s, dAp_dq_s = torch.stack([dApp_dq, dAp_dq])
                dBpp_dq_s, dBp_dq_s = torch.stack([dBpp_dq, dBp_dq])
                
                App, Ap = torch.where(not_converged, torch.stack([Ap, An]), App_s)
                Bpp, Bp = torch.where(not_converged, torch.stack([Bp, Bn]), Bpp_s)
                dApp_dp, dAp_dp = torch.where(not_converged, torch.stack([dAp_dp, dAn_dp]), dApp_dp_s)
                dBpp_dp, dBp_dp = torch.where(not_converged, torch.stack([dBp_dp, dBn_dp]), dBpp_dp_s)
                dApp_dq, dAp_dq = torch.where(not_converged, torch.stack([dAp_dq, dAn_dq]), dApp_dq_s)
                dBpp_dq, dBp_dq = torch.where(not_converged, torch.stack([dBp_dq, dBn_dq]), dBpp_dq_s)

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_a_unscaled = torch.where(swapped, -dI_dq, dI_dp)
                grad_b_unscaled = torch.where(swapped, -dI_dp, dI_dq)
                
                if ctx.needs_input_grad[0]:
                    grad_a[x_intermediate] = grad_a_unscaled
                if ctx.needs_input_grad[1]:
                    grad_b[x_intermediate] = grad_b_unscaled

        if ctx.needs_input_grad[2]:
            grad_x_unscaled = x.pow(a - 1.0) * (1.0 - x).pow(b - 1.0) / _beta(a, b)
            grad_x[x_intermediate] = grad_x_unscaled[x_intermediate]
            
        # --- THIS IS THE FIX ---
        # Apply the chain rule only if the gradient was required.
        # Otherwise, the gradient variable is None, and multiplying it will cause a TypeError.
        grad_a = grad_a * grad_output if ctx.needs_input_grad[0] else None
        grad_b = grad_b * grad_output if ctx.needs_input_grad[1] else None
        grad_x = grad_x * grad_output if ctx.needs_input_grad[2] else None
            
        return grad_a, grad_b, grad_x


def betainc(a, b, x):
    """A clean wrapper for the custom Betainc autograd Function."""
    return Betainc.apply(a, b, x)


def cdf_t(x, df, loc=0.0, scale=1.0):
    """
    Computes the CDF of the Student's t-distribution.
    Differentiable w.r.t. x, df, loc, and scale.
    Now supports tensor inputs.
    """
    if not isinstance(x, torch.Tensor): x = torch.tensor(x)
    if not isinstance(df, torch.Tensor): df = torch.tensor(df)
    if not isinstance(loc, torch.Tensor): loc = torch.tensor(loc)
    if not isinstance(scale, torch.Tensor): scale = torch.tensor(scale)
    
    df, loc, scale = [t.to(x) for t in (df, loc, scale)]
    t = (x - loc) / scale
    x_val = df / (df + t.pow(2))
    
    prob = betainc(
        df / 2.0,
        torch.full_like(df, 0.5),
        x_val
    )
    
    return torch.where(t > 0, 1.0 - 0.5 * prob, 0.5 * prob)



# if __name__ == '__main__':

#     # visual_gradient_comparison.py

#     import torch
#     import matplotlib.pyplot as plt
#     import seaborn as sns


#     # Set a consistent data type and a small epsilon for finite differences
#     DTYPE = torch.float64
#     EPS = 1e-6

#     def verify_gradients(params, param_key_to_check):
#         """
#         Computes and returns both the analytical and numerical gradients for a specific parameter.
        
#         Args:
#             params (dict): Dictionary of all required parameters (x, df, loc, scale).
#             param_key_to_check (str): The key in the params dict for which to compute the gradient.
            
#         Returns:
#             A tuple containing (analytical_gradient, numerical_gradient).
#         """
#         # --- Analytical Gradient (from your custom autograd function) ---
        
#         # Create tensor copies of all parameters, enabling grad for the target parameter
#         t_params = {k: torch.tensor(v, dtype=DTYPE, requires_grad=(k == param_key_to_check)) 
#                     for k, v in params.items()}
        
#         # Run the forward pass
#         cdf_val = cdf_t(**t_params)
        
#         # Run the backward pass to compute gradients
#         cdf_val.backward()
        
#         # Extract the computed gradient
#         analytical_grad = t_params[param_key_to_check].grad.item()
        
#         # --- Numerical Gradient (using the finite difference method) ---
        
#         # Create two versions of the parameters, one with a small positive
#         # perturbation (eps) and one with a negative one.
#         params_plus = params.copy()
#         params_plus[param_key_to_check] += EPS
#         t_params_plus = {k: torch.tensor(v, dtype=DTYPE) for k, v in params_plus.items()}

#         params_minus = params.copy()
#         params_minus[param_key_to_check] -= EPS
#         t_params_minus = {k: torch.tensor(v, dtype=DTYPE) for k, v in params_minus.items()}

#         # Compute the CDF at these two perturbed points
#         cdf_plus = cdf_t(**t_params_plus)
#         cdf_minus = cdf_t(**t_params_minus)
        
#         # The centered finite difference formula: (f(x+h) - f(x-h)) / 2h
#         numerical_grad = (cdf_plus - cdf_minus) / (2 * EPS)

#         return analytical_grad, numerical_grad.item()


#     def plot_gradient_comparison(param_to_vary, fixed_params, param_range):
#         """
#         Generates and displays plots comparing analytical and numerical gradients
#         over a range of values for a single parameter.
#         """
#         print(f"\n📊 Generating plot for parameter: '{param_to_vary}'...")
        
#         analytical_grads = []
#         numerical_grads = []
        
#         # Iterate over the specified range, computing both gradients at each point
#         for val in param_range:
#             current_params = fixed_params.copy()
#             current_params[param_to_vary] = val.item()
            
#             ag, ng = verify_gradients(current_params, param_to_vary)
#             analytical_grads.append(ag)
#             numerical_grads.append(ng)
            
#         # --- Plotting ---
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#         title = f"Gradient Verification for '{param_to_vary}' (other parameters fixed)"
#         fig.suptitle(title, fontsize=16)

#         # Plot 1: Direct comparison of the two gradient calculation methods
#         axes[0].plot(param_range.numpy(), analytical_grads, label='Analytical Gradient (Your Code)', lw=2.5, c='royalblue')
#         axes[0].plot(param_range.numpy(), numerical_grads, label='Numerical Gradient (Finite Diff.)', ls='--', c='darkorange')
#         axes[0].set_xlabel(f"Value of '{param_to_vary}'")
#         axes[0].set_ylabel("Gradient Value")
#         axes[0].set_title(f"∂(CDF) / ∂({param_to_vary})")
#         axes[0].legend()
#         axes[0].grid(True, linestyle=':')
        
#         # Plot 2: Absolute error between the two methods on a log scale
#         abs_error = torch.tensor([abs(a - n) for a, n in zip(analytical_grads, numerical_grads)])
#         axes[1].plot(param_range.numpy(), abs_error.numpy(), c='crimson')
#         axes[1].set_yscale('log')
#         axes[1].set_xlabel(f"Value of '{param_to_vary}'")
#         axes[1].set_ylabel("Absolute Error (log scale)")
#         axes[1].set_title("Error between Analytical and Numerical")
#         axes[1].grid(True, which='both', linestyle=':')
        
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()


#     # Set a professional plotting style
#     sns.set_theme(style="whitegrid")

#     # Define a base set of parameters that will remain fixed while we vary one at a time
#     base_params = {
#         'x': 1.5,
#         'df': 5.0,
#         'loc': 0.5,
#         'scale': 1.2
#     }
#     print("="*60)
#     print("Starting Visual Gradient Verification")
#     print(f"Base parameters (fixed): {base_params}")
#     print("="*60)

#     # --- Generate a plot for each parameter ---

#     # 1. Varying 'x' (the point at which the CDF is evaluated)
#     x_range = torch.linspace(-3.0, 4.0, 100, dtype=DTYPE)
#     plot_gradient_comparison('x', base_params, x_range)
    
#     # 2. Varying 'df' (degrees of freedom)
#     df_range = torch.linspace(2.0, 30.0, 100, dtype=DTYPE)
#     plot_gradient_comparison('df', base_params, df_range)
    
#     # 3. Varying 'loc' (the location or mean of the distribution)
#     loc_range = torch.linspace(-1.0, 2.0, 100, dtype=DTYPE)
#     plot_gradient_comparison('loc', base_params, loc_range)
    
#     # 4. Varying 'scale' (the standard deviation of the distribution)
#     scale_range = torch.linspace(0.5, 3.0, 100, dtype=DTYPE)
#     plot_gradient_comparison('scale', base_params, scale_range)
