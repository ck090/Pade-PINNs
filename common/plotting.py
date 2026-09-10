import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import sys

# Project root (parent of this common/ package), so figs/ always lands next to models/ regardless
# of which PDE's training script imports this module.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Enable TeX for both macOS and Linux
plt.rcParams['text.usetex'] = sys.platform in ('darwin', 'linux')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'

def _hann2_window_np(coord, lo, hi):
    mu = (lo + hi) / 2.0
    sd = (hi - lo) / 2.0
    w = ((1.0 + np.cos(np.pi * (coord - mu) / sd)) / 2.0) ** 2
    return np.where((coord >= lo) & (coord <= hi), w, 0.0)

"""Plot the domain decomposition grid on ground truth and the Hann2 window of an interior subdomain"""
def plot_domain_decomposition(T_plot, X_plot, H_true, subdomains, n_sub_x, n_sub_t, name, folder):
    plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    vmin, vmax = H_true.min(), H_true.max()
    ax.contourf(T_plot, X_plot, H_true, levels=np.linspace(vmin, vmax, 80), cmap='turbo', zorder=-10)
    ax.set_rasterization_zorder(0)
    for sd in subdomains:
        (xl, xh), (tl, th) = sd['overlap_bounds']
        ax.add_patch(plt.Rectangle((tl, xl), th - tl, xh - xl, linewidth=0.9, edgecolor='white', facecolor='none', linestyle='--', alpha=0.7))
        (xl0, xh0), (tl0, th0) = sd['bounds']
        ax.add_patch(plt.Rectangle((tl0, xl0), th0 - tl0, xh0 - xl0, linewidth=1.2, edgecolor='red', facecolor='none', linestyle='-', alpha=0.5))
    ax.set_xlabel(r'$t$', fontweight='bold')
    ax.set_ylabel(r'$x$', fontweight='bold')
    ax.set_title(f'$|u|^2 + |v|^2$ with {n_sub_x}x{n_sub_t} subdomain grid\n(yellow=tile, white-dashed=overlap)', fontsize=8, fontweight='bold')
    ax.set_xlim(T_plot.min(), T_plot.max())
    ax.set_ylim(X_plot.min(), X_plot.max())

    sd_ex = subdomains[(n_sub_t // 2) * n_sub_x + n_sub_x // 2]
    (x_lo_ov, x_hi_ov), (t_lo_ov, t_hi_ov) = sd_ex['overlap_bounds']
    x_sub = np.linspace(x_lo_ov, x_hi_ov, 200)
    t_sub = np.linspace(t_lo_ov, t_hi_ov, 200)
    X_sub, T_sub = np.meshgrid(x_sub, t_sub)
    window = _hann2_window_np(X_sub, x_lo_ov, x_hi_ov) * _hann2_window_np(T_sub, t_lo_ov, t_hi_ov)

    ax = axes[1]
    im = ax.pcolormesh(T_sub, X_sub, window, cmap='plasma', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hann2 weight')
    ax.set_title(f'Hann2 window - subdomain [t={sd_ex["it"]}, x={sd_ex["ix"]}]\nx,t normalised to [-1, 1] within overlap region', fontsize=8, fontweight='bold')
    ax.set_xlabel(r'$t$', fontweight='bold')
    ax.set_ylabel(r'$x$', fontweight='bold')

    plt.tight_layout()
    save_dir = os.path.join(BASE_DIR, "figs")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()

"""Plotting helper function"""
def plot_prediction(T_plot, X_plot, U_pred, U_true, X_bc_left, X_bc_right, params, intervals, name, folder):
    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif', 'text.latex.preamble': ''})
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 0.8], wspace=0.25)

    gs_left = gs[0].subgridspec(2, 1, hspace=0.3)
    gs_right = gs[1].subgridspec(3, 1, hspace=0.3, height_ratios=[0.5, 0.5, 0.5])

    # 5 evenly spaced ticks spanning whatever the domain is (e.g. [-5,5] or [-10,10]) - a tick count, not a domain bound
    y_ticks = np.linspace(X_plot.min(), X_plot.max(), 5)

    # --- Row 1 Left: Prediction ---
    ax1 = fig.add_subplot(gs_left[0])
    vmin = min(U_pred.min(), U_true.min())
    vmax = max(U_pred.max(), U_true.max())
    levels = np.linspace(vmin, vmax, 100)

    im1 = ax1.contourf(T_plot, X_plot, U_pred, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax, zorder=-10)
    ax1.set_rasterization_zorder(0)

    if isinstance(params, dict):
        param_str = ", ".join([rf"${k}={v:.5f}$" for k, v in params.items()])
        title_str = rf"Prediction $\hat u(x,t)$ for {param_str}"
    elif isinstance(params, (list, tuple, np.ndarray)):
        param_str = ", ".join([f"{k:.5f}" for k in params])
        title_str = rf"Prediction $\hat u(x,t)$ for = \{{ {param_str} \}}"
    else:
        title_str = rf"Prediction $\hat u(x,t)$ for = {params:.5f}"

    ax1.set_title(title_str, fontweight='bold', pad=4)
    ax1.set_ylabel(r'$x$', fontweight='bold')
    # ax1.set_xlabel(r'$t$', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.01, fraction=0.03, format='%.2f')
    cbar1.ax.tick_params(labelsize=6)
    cbar1.solids.set_rasterized(True)
    cbar1.solids.set_edgecolor("face")

    ax1.set_xlim(T_plot.min(), T_plot.max())
    ax1.set_ylim(X_plot.min(), X_plot.max())
    ax1.set_yticks(y_ticks)

    # --- Row 2 Left: Ground Truth ---
    ax2 = fig.add_subplot(gs_left[1])
    im2 = ax2.contourf(T_plot, X_plot, U_true, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax, zorder=-10)
    ax2.set_rasterization_zorder(0)

    if isinstance(params, dict):
        param_str = ", ".join([rf"${k}={v:0.5f}$" for k, v in params.items()])
        title_str = rf"Ground Truth $u(x,t)$ for {param_str}"
    elif isinstance(params, (list, tuple, np.ndarray)):
        param_str = ", ".join([f"{k:0.5f}" for k in params])
        title_str = rf"Ground Truth $u(x,t)$ for = \{{ {param_str} \}}"
    else:
        title_str = rf"Ground Truth $u(x,t)$ for = {params:0.5f}"

    ax2.set_title(title_str, fontweight='bold', pad=4)
    ax2.set_ylabel(r'$x$', fontweight='bold')
    ax2.set_xlabel(r'$t$', fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.01, fraction=0.03, format='%.2f')
    cbar2.ax.tick_params(labelsize=6)
    cbar2.solids.set_rasterized(True)
    cbar2.solids.set_edgecolor("face")

    # Scatter boundary condition points on ground truth
    ax2.scatter(X_bc_left[:, 1], X_bc_left[:, 0], c='black', s=6, marker='x', linewidths=0.7, zorder=10, clip_on=False)
    ax2.scatter(X_bc_right[:, 1], X_bc_right[:, 0], c='black', s=6, marker='x', linewidths=0.7, zorder=10, clip_on=False)
    ax2.set_xlim(T_plot.min(), T_plot.max())
    ax2.set_ylim(X_plot.min(), X_plot.max())
    ax2.set_yticks(y_ticks)

    # --- Right Column: Comparison ---
    # Convert time points to data indices
    unique_times = np.unique(T_plot)

    idx_1 = np.argmin(np.abs(unique_times - intervals[0]))
    ax3 = fig.add_subplot(gs_right[0])
    ax3.plot(X_plot[idx_1], U_true[idx_1], label=f'GT', lw=1.6)
    ax3.plot(X_plot[idx_1], U_pred[idx_1], '--r', label=f'Pred', lw=1.6)
    ax3.set_ylabel(rf"@ $t={intervals[0]:.2f}$", fontweight='bold')
    # ax3.set_ylabel(r'$u(x,t)$', fontweight='bold')
    ax3.set_title(r'Comparison at Selected Time Points', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.7)

    idx_2 = np.argmin(np.abs(unique_times - intervals[1]))
    ax4 = fig.add_subplot(gs_right[1])
    ax4.plot(X_plot[idx_2], U_true[idx_2], label=f'GT', lw=1.6)
    ax4.plot(X_plot[idx_2], U_pred[idx_2], '--r', label=f'Pred', lw=1.6)
    ax4.set_ylabel(rf"@ $t={intervals[1]:.2f}$", fontweight='bold')
    # ax4.set_ylabel(r'$u(x,t)$', fontweight='bold')
    ax4.grid(True, alpha=0.7)

    idx_3 = np.argmin(np.abs(unique_times - intervals[2]))
    ax5 = fig.add_subplot(gs_right[2])
    ax5.plot(X_plot[idx_3], U_true[idx_3], label=f'GT', lw=1.6)
    ax5.plot(X_plot[idx_3], U_pred[idx_3], '--r', label=f'Pred', lw=1.6)
    ax5.set_xlabel(r'$x$', fontweight='bold')
    ax5.set_ylabel(rf"@ $t={intervals[2]:.2f}$", fontweight='bold')
    # ax5.set_ylabel(r'$u(x,t)$', fontweight='bold')
    ax5.grid(True, alpha=0.7)

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', ncol=1, fontsize=6, bbox_to_anchor=(0.91, 0.5))
    for ax in [ax3, ax4, ax5]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # fig.align_labels()
    save_dir = os.path.join(BASE_DIR, "figs", folder) if folder else os.path.join(BASE_DIR, "figs")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()


"""Animation helper function"""
def animate_prediction(T_plot, X_plot, U_pred, U_true, params, name, folder, fps=10):
    plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

    vmin = min(U_pred.min(), U_true.min())
    vmax = max(U_pred.max(), U_true.max())

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    ax.set_xlim(X_plot.min(), X_plot.max())
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r'$x$', fontweight='bold')
    ax.set_ylabel(r'$u(x,t)$', fontweight='bold')
    ax.grid(True, alpha=0.3)
    line_pred, = ax.plot([], [], 'b-', lw=2, label='Prediction')
    line_true, = ax.plot([], [], 'r--', lw=2, label='Ground Truth')
    ax.legend(loc='upper right')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    # Select frames at every 0.05 seconds
    times = T_plot[:, 0]
    frame_indices = []
    last_time = -float('inf')
    for i, t in enumerate(times):
        if t >= last_time + 0.05:
            frame_indices.append(i)
            last_time = t

    def init():
        line_pred.set_data([], [])
        line_true.set_data([], [])
        time_text.set_text('')
        return line_pred, line_true, time_text

    def animate(frame_idx):
        frame = frame_indices[frame_idx]
        t = T_plot[frame, 0]
        line_pred.set_data(X_plot[frame], U_pred[frame])
        line_true.set_data(X_plot[frame], U_true[frame])

        if isinstance(params, dict):
            param_str = ", ".join([rf"${k}={v:.4f}$" for k, v in params.items()])
        elif isinstance(params, (list, tuple, np.ndarray)):
            param_str = ", ".join([f"{k:.4f}" for k in params])
        else:
            param_str = f"{params:.4f}"

        time_text.set_text(rf'$t = {t:.3f}$, {param_str}')
        return line_pred, line_true, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(frame_indices), interval=1000/fps, blit=True)
    save_dir = os.path.join(BASE_DIR, "figs", folder) if folder else os.path.join(BASE_DIR, "figs")
    os.makedirs(save_dir, exist_ok=True)
    if (sys.platform == "darwin"):
        anim.save(os.path.join(save_dir, f"{name}_animation.mov"), fps=fps, dpi=300)
        anim.save(os.path.join(save_dir, f"{name}_animation.gif"), fps=fps, dpi=300)
    elif (sys.platform.startswith("linux")):
        anim.save(os.path.join(save_dir, f"{name}_animation.gif"), fps=fps, dpi=300)
    else:
        print("OS not recognized; cannot save animation.")
    plt.close()


"""Per-timestep relative L2 error for a vanilla PINN (single curve, no Pade comparison)"""
def plot_pinn_error_evolution(t_eval, errs, params, name, folder):
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    avg = np.mean(errs)

    if isinstance(params, dict):
        param_str = ", ".join([rf"${k}={v:.4f}$" for k, v in params.items()])
    elif isinstance(params, (list, tuple, np.ndarray)):
        param_str = ", ".join([f"{k:.4f}" for k in params])
    else:
        param_str = f"{params:.4f}"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t_eval, errs, lw=2, color='steelblue', label='PINN')
    ax.axhline(avg, color='steelblue', linestyle=':', alpha=0.7, label=f'Mean PINN  {avg:.2e}')
    ax.set_title(f'Relative L2 error over time ({param_str})')
    ax.set_xlabel('t')
    ax.set_ylabel('Relative L2 error')
    ax.legend()
    plt.tight_layout()

    save_dir = os.path.join(BASE_DIR, "figs", folder, "error") if folder else os.path.join(BASE_DIR, "figs", "error")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}_error.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(save_dir, f"{name}_error.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()


"""Plot training loss (residual + BC) evolution over epochs"""
def plot_loss_history(epochs_logged, loss_res_hist, loss_bc_hist, name, folder):
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(epochs_logged, loss_res_hist, label='Residual loss', lw=2, color='steelblue')
    ax.semilogy(epochs_logged, loss_bc_hist,  label='BC loss', lw=2, linestyle='--', color='tab:red')
    ax.set_title('Training loss over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()

    save_dir = os.path.join(BASE_DIR, "figs", folder, "loss") if folder else os.path.join(BASE_DIR, "figs", "loss")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}_loss.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(save_dir, f"{name}_loss.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()


"""Plot per-timestep error evolution helper function"""
def plot_error_evolution(t_eval, errs_pade, errs_nn, params, name, folder):
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

    avg_pade = np.mean(errs_pade)
    avg_nn = np.mean(errs_nn)

    if isinstance(params, dict):
        param_str = ", ".join([rf"${k}={v:.4f}$" for k, v in params.items()])
    elif isinstance(params, (list, tuple, np.ndarray)):
        param_str = ", ".join([f"{k:.4f}" for k in params])
    else:
        param_str = f"{params:.4f}"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t_eval, errs_pade, label='Pade only',   lw=2,   linestyle='--', color='coral')
    ax.semilogy(t_eval, errs_nn,   label='Pade + NN',   lw=2,   color='steelblue')
    ax.axhline(avg_pade, color='coral',     linestyle=':', alpha=0.7, label=f'Mean Pade  {avg_pade:.2e}')
    ax.axhline(avg_nn,   color='steelblue', linestyle=':', alpha=0.7, label=f'Mean Pade+NN {avg_nn:.2e}')
    ax.set_title(f'Relative L2 error over time ({param_str})')
    ax.set_xlabel('t')
    ax.set_ylabel('Relative L2 error')
    ax.legend()
    plt.tight_layout()

    save_dir = os.path.join(BASE_DIR, "figs", "error")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}_error.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(save_dir, f"{name}_error.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
