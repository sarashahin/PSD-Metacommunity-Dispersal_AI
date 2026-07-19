############################################
# analysis.py
############################################
"""
Analysis routines for multi–patch (spatial) models with dispersal:
 - Standard plots (total biomass, histograms, covariance, etc.)
 - Additional visualizations for spatial distributions and dispersal flux
 - Diagnostics to quantify spatial variation across patches.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
from utils import mav, bav, mean_se

logger = logging.getLogger(__name__)

def alpha_diversity(trajectory, threshold=None):
    if threshold is None:
        from config import THRESHOLD
        threshold = THRESHOLD
    presence = (trajectory > threshold).astype(int)
    total_presence = np.sum(presence, axis=(1,2,3))
    return total_presence

def turnover_rate(trajectory, recording_step):
    presence = (trajectory > 0).astype(int)
    diffs = np.abs(np.diff(presence, axis=0))
    changes = np.sum(diffs, axis=(1,2,3))
    rate = changes / float(recording_step)
    return rate

def plot_total_biomasses(trajectories, labels, title, filename):
    plt.figure(figsize=(10,6))
    for traj, label in zip(trajectories, labels):
        total_biomass = np.sum(traj, axis=(1,2,3))
        plt.plot(total_biomass, label=label)
    plt.xlabel("Time Index")
    plt.ylabel("Total Biomass")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def plot_trajectories(trajectories, labels, title, filename_base):
    for traj, label in zip(trajectories, labels):
        plt.figure(figsize=(10,6))
        avg_biomass = np.mean(traj, axis=(1,2))
        for i in range(avg_biomass.shape[1]):
            plt.plot(avg_biomass[:, i], alpha=0.7)
        plt.xlabel("Time Index")
        plt.ylabel("Average Biomass (per species)")
        plt.title(label)
        fname = label + "_" + filename_base
        plt.savefig(fname)
        plt.close()
        logger.info(f"Saved figure {fname}")

def histogram_biomass(trajectories, labels, filename):
    plt.figure(figsize=(10,6))
    for traj, label in zip(trajectories, labels):
        data = traj.flatten()
        data = data[data >= 1e-12]
        if len(data) == 0:
            continue
        plt.hist(np.log10(data), bins=50, alpha=0.4, label=label, density=True)
    plt.xlabel("log10(Biomass)")
    plt.ylabel("Density")
    plt.title("Biomass Distribution")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def covariance_matrix_plot(trajectory, title, filename):
    half = trajectory.shape[0] // 2
    portion = trajectory[half:, ...].reshape(trajectory.shape[0]-half, -1)
    covmat = np.cov(portion.T)
    plt.figure(figsize=(6,5))
    plt.imshow(covmat, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Covariance')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def plot_spatial_heatmap(trajectory, time_index, species_index=0, title=None, filename=None):
    spatial_data = trajectory[time_index, :, :, species_index]
    plt.figure(figsize=(8,6))
    plt.imshow(spatial_data, cmap="viridis", origin="lower")
    plt.colorbar(label="Biomass")
    if title is None:
        title = f"Spatial Distribution (Species {species_index}) at Record {time_index}"
    plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved spatial heatmap to {filename}")
    else:
        plt.show()

def animate_spatial_evolution(trajectory, species_index=0, interval=200, save_filename=None):
    fig, ax = plt.subplots(figsize=(8,6))
    n_records = trajectory.shape[0]
    im = ax.imshow(trajectory[0, :, :, species_index], cmap="viridis", origin="lower")
    ax.set_title(f"Time Record: 0")
    plt.colorbar(im, ax=ax, label="Biomass")
    
    def update(frame):
        im.set_data(trajectory[frame, :, :, species_index])
        ax.set_title(f"Time Record: {frame}")
        return im,
    
    ani = animation.FuncAnimation(fig, update, frames=n_records, interval=interval, blit=True)
    if save_filename:
        ani.save(save_filename, writer="ffmpeg")
        logger.info(f"Saved animation to {save_filename}")
    else:
        plt.show()
    return ani

def verify_spatial_variation(trajectory, species_index=0, plot_std=False, output_prefix="VariationCheck"):
    n_records = trajectory.shape[0]
    std_values = np.zeros(n_records, dtype=float)
    
    for t in range(n_records):
        data_t = trajectory[t, :, :, species_index]
        min_val = np.min(data_t)
        max_val = np.max(data_t)
        mean_val = np.mean(data_t)
        std_val = np.std(data_t)
        std_values[t] = std_val
        logger.info(f"[Species {species_index}] Time {t}: min={min_val:.4g}, max={max_val:.4g}, mean={mean_val:.4g}, std={std_val:.4g}")
    
    if plot_std:
        plt.figure(figsize=(8,5))
        plt.plot(std_values, marker='o', linestyle='-')
        plt.title(f"Spatial STD Over Time (Species {species_index})")
        plt.xlabel("Time Record Index")
        plt.ylabel("Standard Deviation (Biomass)")
        outname = f"{output_prefix}_species_{species_index}.png"
        plt.savefig(outname)
        plt.close()
        logger.info(f"Saved standard deviation plot to {outname}")
        

def plot_dispersal_flux_heatmap(flux_trajectory, time_index, species_index=0, title=None, filename=None):
    spatial_data = flux_trajectory[time_index, :, :, species_index]
    plt.figure(figsize=(8,6))
    plt.imshow(spatial_data, cmap="plasma", origin="lower")
    plt.colorbar(label="Dispersal Flux")
    if title is None:
        title = f"Dispersal Flux (Species {species_index}) at Record {time_index}"
    plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved dispersal flux heatmap to {filename}")
    else:
        plt.show()

def animate_dispersal_flux(flux_trajectory, species_index=0, interval=200, save_filename=None):
    fig, ax = plt.subplots(figsize=(8,6))
    n_records = flux_trajectory.shape[0]
    im = ax.imshow(flux_trajectory[0, :, :, species_index], cmap="plasma", origin="lower")
    ax.set_title(f"Time Record: 0")
    plt.colorbar(im, ax=ax, label="Dispersal Flux")
    
    def update(frame):
        im.set_data(flux_trajectory[frame, :, :, species_index])
        ax.set_title(f"Time Record: {frame}")
        return im,
    
    ani = animation.FuncAnimation(fig, update, frames=n_records, interval=interval, blit=True)
    if save_filename:
        ani.save(save_filename, writer="ffmpeg")
        logger.info(f"Saved dispersal flux animation to {save_filename}")
    else:
        plt.show()
    return ani

def save_model_output(ibm_trajectory, psd_trajectory,dispersal_flux_traj, psd2_trajectory, psd2_waiting,
                      psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob,
                      psd2_flux,  # <-- new: dispersal flux for PSD2
                      ode_trajectory):
    np.savez(
        "model_outputs.npz",
        IBM=ibm_trajectory,
        PSD=psd_trajectory,
        PSD_flux_traj=dispersal_flux_traj,
        PSD2=psd2_trajectory,
        PSD2_waiting=psd2_waiting,
        PSD2_poisson_clock=psd2_poisson_clock,
        PSD2_growth_rate=psd2_growth_rate,
        PSD2_invasion_rate=psd2_invasion_rate,
        PSD2_est_prob=psd2_est_prob,
        PSD2_flux=psd2_flux,  # <-- added dispersal flux from PSD2
        ODE=ode_trajectory
    )
    logger.info("Saved model outputs to 'model_outputs.npz' for further analysis.")
