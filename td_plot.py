import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# --- Configuration ---
# Directory for input CSVs and output PDF
DATA_DIR = 'plots'

# Set the prefixes for your two experiment runs
EXP1_PREFIX = 'SAC_19'
EXP2_PREFIX = 'SAC_23' # <-- Updated as requested

# Set the maximum training step to plot
MAX_STEP = 3_124_000

# Plotting settings
OUTPUT_FILENAME = "sac_sequential_plots.pdf"
SMOOTHING_WINDOW = 20

# --- LaTeX Plotting Setup ---
def setup_latex_plotting_params():
    """Sets matplotlib parameters for LaTeX-friendly plots."""
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.autolayout": True,
        "savefig.dpi": 300
    })

def get_sequential_figsize():
    """Calculates a tall figure size suitable for a 4x1 layout."""
    textwidth_pt = 427.43153
    pt_to_inch = 1.0 / 72.27
    textwidth_in = textwidth_pt * pt_to_inch
    
    # Use the full text width for the figure width
    fig_width_in = textwidth_in * 1.0
    # Make the figure tall enough to accommodate 4 sequential plots
    fig_height_in = fig_width_in * 1.5
    
    return (fig_width_in, fig_height_in)

# --- Helper Function for Plotting a Single Run ---
def plot_single_run(ax, exp_prefix, metric_tag, color, label):
    """Loads a single CSV, filters by MAX_STEP, and plots its data."""
    filename = f"run-{exp_prefix}-tag-{metric_tag}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Warning: File not found, skipping: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
        df['Step'] = pd.to_numeric(df['Step'])
        df['Value'] = pd.to_numeric(df['Value'])
        
        # Filter the DataFrame before plotting
        df = df[df['Step'] <= MAX_STEP].copy()
        
        ax.plot(df['Step'], df['Value'], color=color, alpha=0.2, linewidth=1)
        smoothed_value = df['Value'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        ax.plot(df['Step'], smoothed_value, color=color, label=label, linewidth=2.0)
        
    except Exception as e:
        print(f"Could not read or plot file {filepath}: {e}")

# --- Individual Plotting Functions ---
def plot_critic_loss(ax, exp1, exp2, colors):
    metric_tag = 'train_critic_loss'
    plot_single_run(ax, exp1, metric_tag, colors[0], exp1)
    plot_single_run(ax, exp2, metric_tag, colors[1], exp2)
    ax.set_title('Critic Loss')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_actor_loss(ax, exp1, exp2, colors):
    metric_tag = 'train_actor_loss'
    plot_single_run(ax, exp1, metric_tag, colors[0], exp1)
    plot_single_run(ax, exp2, metric_tag, colors[1], exp2)
    ax.set_title('Actor Loss')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_ep_reward(ax, exp1, exp2, colors):
    metric_tag = 'rollout_ep_rew_mean'
    plot_single_run(ax, exp1, metric_tag, colors[0], exp1)
    plot_single_run(ax, exp2, metric_tag, colors[1], exp2)
    ax.set_title('Mean Episode Reward')
    ax.set_ylabel('Reward')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_ep_length(ax, exp1, exp2, colors):
    metric_tag = 'rollout_ep_len_mean'
    plot_single_run(ax, exp1, metric_tag, colors[0], exp1)
    plot_single_run(ax, exp2, metric_tag, colors[1], exp2)
    ax.set_title('Mean Episode Length')
    ax.set_ylabel('Steps')
    ax.grid(True, linestyle='--', alpha=0.6)

# --- Main Function to Orchestrate Plotting ---
def main():
    """Creates the main sequential figure and coordinates all plotting."""
    setup_latex_plotting_params()
    fig_size = get_sequential_figsize()
    
    # Define colors for consistent plotting
    colors = ['#1f77b4', '#ff7f0e']
    
    # Create a 4x1 subplot grid with a shared x-axis
    fig, axs = plt.subplots(4, 1, figsize=fig_size, sharex=True)

    # Call the dedicated function for each subplot
    plot_ep_length(axs[0], EXP1_PREFIX, EXP2_PREFIX, colors)
    plot_ep_reward(axs[1], EXP1_PREFIX, EXP2_PREFIX, colors)
    plot_actor_loss(axs[2], EXP1_PREFIX, EXP2_PREFIX, colors)
    plot_critic_loss(axs[3], EXP1_PREFIX, EXP2_PREFIX, colors)
    
    # --- Final Figure Adjustments ---
    # Set the shared x-axis label only on the bottom-most plot
    axs[-1].set_xlabel("Training Step")
    axs[-1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Explicitly set the x-axis limit for all shared axes
    axs[-1].set_xlim(left=0, right=MAX_STEP)
    
    # CHANGED: Add the legend to the first subplot.
    # 'best' location automatically finds a good spot.
    axs[0].legend(loc='best')
    
    # CHANGED: Use simpler tight_layout for better spacing between subplots.
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save the final figure
    output_path = os.path.join(DATA_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved successfully as '{output_path}'")

if __name__ == '__main__':
    main()