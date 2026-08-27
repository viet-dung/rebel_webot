import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import SAC
from tqdm import tqdm
import time

# --- IMPORT YOUR ENVIRONMENT FILE ---
# You can swap this to RealEnv if you want to run this test on the physical robot
from simulationEnv import SimulationEnv 
from realEnv import RealEnv

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# --- CORE PARAMETERS ---
NUM_EVAL_EPISODES = 1     # Number of distinct starting positions to test
SUCCESS_THRESHOLD = 0.01    # Distance in meters for success

# --- PATHS AND FILENAMES ---
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CODE_DIR, "sac_stack_best", "SAC_reaching_4130000_steps.zip") 
OUTPUT_DIR = "policy_comparison_results"
CSV_FILENAME = "sac_vs_random_raw_data.csv"
LATEX_TABLE_FILENAME = "sac_vs_random_summary_table.tex"
SUMMARY_PLOT_FILENAME = "sac_vs_random_summary_plots.png"

# ==============================================================================
# EVALUATION FUNCTION (Provided by you, used directly)
# ==============================================================================

# ==============================================================================
# LATEX PLOTTING CONFIGURATION
# ==============================================================================
USE_LATEX = True # Master switch to enable/disable LaTeX-style plots

def configure_latex_plotting():
    """Sets the global matplotlib rcParams for high-quality, LaTeX-friendly plots."""
    if not USE_LATEX:
        print("INFO: Standard Matplotlib rendering is enabled.")
        return

    from matplotlib import rcParams
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"], # A good, standard serif font
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.autolayout": True, # Prevents labels from being cut off
        "savefig.dpi": 300,
        "text.usetex": False # Set to True if you have a full LaTeX installation
    })
    print("INFO: LaTeX-friendly plotting is ENABLED.")

def get_latex_figsize(width_scale=1.0, height_ratio=None):
    """
    Calculates the figure size based on LaTeX text width.
    :param width_scale: Fraction of the text width the plot should occupy.
    :param height_ratio: Aspect ratio (height/width). Defaults to the golden ratio.
    """
    if not USE_LATEX:
        return (10, 6) # Return a default size if not using LaTeX

    textwidth_pt = 427.43153
    pt_to_inch = 1.0 / 72.27
    textwidth_in = textwidth_pt * pt_to_inch
    
    fig_width_in = textwidth_in * width_scale
    
    if height_ratio is None:
        golden_ratio = (np.sqrt(5) - 1.0) / 2.0
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = fig_width_in * height_ratio
        
    return (fig_width_in, fig_height_in)

def evaluate_policy(env, policy_name, model=None, num_episodes=10, box_poses_to_use=None):
    all_episode_data = []
    generated_box_poses = []
    is_forced_pose_run = box_poses_to_use is not None

    for i in tqdm(range(num_episodes), desc=f"Evaluating {policy_name} Policy"):
        obs, _ = env.reset()
        
        # If a list of poses is provided, force the environment to use them
        if is_forced_pose_run:
            env.current_box_pose = np.array(box_poses_to_use[i])
            env.set_box_pose = False # Tell the env not to generate a new random pose
            obs = env.get_observation() # Get a fresh observation with the forced pose
        
        # Always save the pose that was actually used for the trial
        generated_box_poses.append(env.current_box_pose.copy())
        
        initial_gripper_pos = obs['achieved_goal'][0:3].copy()
        initial_box_pos = obs['achieved_goal'][7:10].copy()
        initial_dist = np.linalg.norm(initial_box_pos - initial_gripper_pos)

        terminated, truncated = False, False
        steps_taken = 0
        while not terminated and not truncated:
            # Use model if provided (for SAC), otherwise sample a random action
            action, _ = (model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None))
            obs, _, terminated, truncated, _ = env.step(action)
            steps_taken += 1
        
        is_success = terminated
        final_gripper_pos = obs['achieved_goal'][0:3].copy()
        final_box_pos = obs['achieved_goal'][7:10].copy()
        final_dist = np.linalg.norm(final_box_pos - final_gripper_pos)
        
        all_episode_data.append({
            'policy': policy_name,
            'initial_dist': initial_dist,
            'final_dist': final_dist,
            'is_success': is_success,
            'steps_taken': steps_taken,
        })
    return all_episode_data, generated_box_poses

# ==============================================================================
# PLOTTING AND TABLE GENERATION (Adapted for SAC vs. Random)
# ==============================================================================

def create_success_scatter_plot(df, filename):
    """
    Creates a single, combined scatter plot visualizing the four possible Sim2Real outcomes,
    formatted for inclusion in a LaTeX document.
    """
    if 'Simulation' not in df['environment'].unique() or 'Real World' not in df['environment'].unique():
        print("Cannot create scatter plot: Data for both 'Simulation' and 'Real World' is required.")
        return

    # Pivot data to get Sim and Real outcomes on the same row
    pivot_df = df.pivot(index='episode', columns='environment', values=['initial_box_x', 'initial_box_y', 'is_success'])

    def determine_outcome_category(row):
        sim_success = row[('is_success', 'Simulation')]
        real_success = row[('is_success', 'Real World')]
        if sim_success and real_success: return 'Success (Sim & Real)'
        elif sim_success and not real_success: return 'Sim Success, Real Fail'
        elif not sim_success and real_success: return 'Real Success, Sim Fail'
        else: return 'Failure (Sim & Real)'

    pivot_df['outcome'] = pivot_df.apply(determine_outcome_category, axis=1)

    # Define visual styles for each outcome
    styles = {
        'Success (Sim & Real)':   {'marker': 'o', 'color': 'green',   'label': 'Success (Both)'},
        'Failure (Sim & Real)':   {'marker': 'X', 'color': 'red',     'label': 'Failure (Both)'},
        'Sim Success, Real Fail': {'marker': 'v', 'color': 'blue',    'label': 'Sim Success Only'},
        'Real Success, Sim Fail': {'marker': '^', 'color': 'magenta', 'label': 'Real Success Only'}
    }
    
    # --- KEY CHANGE: Use the LaTeX figure size ---
    # A square-ish aspect ratio is good for this plot.
    fig, ax = plt.subplots(figsize=(12,10))

    for outcome, style in styles.items():
        group = pivot_df[pivot_df['outcome'] == outcome]
        if not group.empty:
            ax.scatter(group[('initial_box_x', 'Simulation')], 
                       group[('initial_box_y', 'Simulation')],
                       marker=style['marker'], color=style['color'], label=style['label'],
                       alpha=0.8, s=60, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Box X-coordinate (m)')
    ax.set_ylabel('Box Y-coordinate (m)')
    ax.legend(title='Outcome', loc="best")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Save as PDF for best quality in LaTeX documents
    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"LaTeX-formatted scatter plot saved to {pdf_filename}")
     

def plot_real_success_rate(df, filename_base):
    """Generates a single bar chart for the success rate in the Real World ONLY."""
    
    # Filter for only the real-world data
    real_df = df[df['environment'] == 'Real World'].copy()
    if real_df.empty:
        print("Skipping real-world success rate plot: No real-world data found.")
        return

    success_rate = real_df['is_success'].mean() * 100

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar('SAC Policy', success_rate, color='darkorange', width=0.5, edgecolor='black')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the percentage text on the bar
    ax.text(0, success_rate, f'{success_rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=0.3"))
    output_filename = filename_base + "_real_success_rate.pdf"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Real-world success rate plot saved to {output_filename}")
     


def plot_real_relative_distance(df, filename_base):
    """Generates a box plot for the relative distance progress in the Real World ONLY."""
    
    real_df = df[df['environment'] == 'Real World'].copy()
    if real_df.empty:
        print("Skipping real-world relative distance plot: No real-world data found.")
        return

    # Calculate progress percentage for the real-world trials
    real_df['progress_perc'] = 100 * (real_df['initial_dist'] - real_df['final_dist']) / real_df['initial_dist'].replace(0, 1e-6)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=real_df, y='progress_perc', color='darkorange', ax=ax, width=0.4)
    
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='100% Progress (Goal)')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='0% Progress')
    
    ax.set_ylabel('Progress Toward Goal (%)', fontsize=12)
    ax.set_xlabel('SAC Policy', fontsize=12)
    ax.legend(loc="best")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_filename = filename_base + "_real_relative_distance.pdf"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Real-world relative distance plot saved to {output_filename}")
     


def plot_real_absolute_distance(df, filename_base):
    """Generates a box plot for the final absolute distance in the Real World ONLY."""
    
    real_df = df[df['environment'] == 'Real World'].copy()
    if real_df.empty:
        print("Skipping real-world absolute distance plot: No real-world data found.")
        return
        
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=real_df, y='final_dist', color='darkorange', ax=ax, width=0.4)
    
    ax.axhline(SUCCESS_THRESHOLD, color='green', linestyle='--', linewidth=1.5, label=f'Success Threshold ({SUCCESS_THRESHOLD}m)')
    ax.set_yscale('log')
    
    ax.set_ylabel('Final Distance (meters, log scale)', fontsize=12)
    ax.set_xlabel('SAC Policy', fontsize=12)
    ax.legend(loc="best")
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.7)

    output_filename = filename_base + "_real_absolute_distance.pdf"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Real-world absolute distance plot saved to {output_filename}")
     

def create_detailed_latex_table(df, filename):
    """
    Creates a detailed LaTeX table with a more informative outcome column.
    - ✓: Success in both Sim and Real.
    - S: Success in Sim only (failed in Real).
    - R: Success in Real only (failed in Sim).
    - ✗: Failure in both.

    Requires the 'pifont' package in your LaTeX document.
    """
    # Ensure data exists for both environments to make a comparison
    if 'Simulation' not in df['environment'].unique() or 'Real World' not in df['environment'].unique():
        print("Cannot create detailed LaTeX table: Data for both 'Simulation' and 'Real World' is required.")
        return
        
    # Pivot the table to get Sim and Real results on the same row for each episode
    pivot_df = df.pivot(index='episode', columns='environment', values=['initial_box_x', 'initial_box_y', 'is_success'])
    
    # Build the final table for export
    result_df = pd.DataFrame(index=pivot_df.index)
    
    # Coordinates are the same, so we take them from the Sim run
    result_df['Start X (m)'] = pivot_df[('initial_box_x', 'Simulation')].map('{:.3f}'.format)
    result_df['Start Y (m)'] = pivot_df[('initial_box_y', 'Simulation')].map('{:.3f}'.format)
    
    # --- NEW DETAILED OUTCOME LOGIC ---
    def determine_detailed_outcome(row):
        sim_success = row[('is_success', 'Simulation')]
        real_success = row[('is_success', 'Real World')]
        
        if sim_success and real_success:
            return r'\cmark'  # Success in both
        elif sim_success and not real_success:
            return r'\textbf{S}'  # Succeeded in Sim only
        elif not sim_success and real_success:
            return r'\textbf{R}'  # Succeeded in Real only
        else: # Both failed
            return r'\xmark'

    # Apply the new logic function to create the single outcome column
    result_df['Outcome'] = pivot_df.apply(determine_detailed_outcome, axis=1)
    
    result_df = result_df.rename_axis('Trial ID')
    
    # Generate the LaTeX string with an updated caption explaining the symbols
    latex_string = result_df.to_latex(
        index=True,
        escape=False,
        column_format='l c c c', 
        header=['Start X (m)', 'Start Y (m)', 'Outcome'],
        caption='Detailed Sim2Real outcome. \\cmark: Success in both. \\textbf{S}: Success in Simulation only. \\textbf{R}: Success in Real World only. \\xmark: Failure in both.',
        label='tab:detailed_sim2real_outcome'
    )
    
    # Write the string to a .tex file
    with open(filename, 'w') as f:
        f.write(r'%% Add to your LaTeX preamble: \usepackage{pifont} \newcommand{\cmark}{\ding{51}} \newcommand{\xmark}{\ding{55}}' + '\n\n')
        f.write(latex_string)
        
    print(f"Detailed-outcome LaTeX results table saved to {filename}")

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model file not found at '{MODEL_PATH}'"); return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    used_box_poses = []
    
    # --- PHASE 1: Run SAC on Simulation ---
    print("\n--- PHASE 1: Evaluating SAC Policy on Simulation ---")
    sim_env = None
    try:
        sim_env = SimulationEnv(port='1244', env_id=10)
        sac_model = SAC.load(MODEL_PATH, env=sim_env)
        sim_results, used_box_poses = evaluate_policy(
            env=sim_env, policy_name="SAC", model=sac_model, 
            num_episodes=NUM_EVAL_EPISODES
        )
        for i, result in enumerate(sim_results):
            result['environment'] = 'Simulation'
            result['episode'] = i
            result['initial_box_x'] = used_box_poses[i][0]
            result['initial_box_y'] = used_box_poses[i][1]
        all_results.extend(sim_results)
    finally:
        if sim_env: sim_env.close()
        print("--- Simulation Phase Finished ---")

    # --- PHASE 2: Run SAC on Real Robot ---
    if used_box_poses:
        print("\n--- PHASE 2: Evaluating SAC Policy on Real Robot ---")
        real_env = None
        try:
            real_env = RealEnv(port='1244', env_id=10)
            sac_model = SAC.load(MODEL_PATH, env=real_env)
            real_results, _ = evaluate_policy(
                env=real_env, policy_name="SAC", model=sac_model,
                num_episodes=NUM_EVAL_EPISODES, box_poses_to_use=used_box_poses
            )
            for i, result in enumerate(real_results):
                result['environment'] = 'Real World'
                result['episode'] = i
                result['initial_box_x'] = used_box_poses[i][0]
                result['initial_box_y'] = used_box_poses[i][1]
            all_results.extend(real_results)
        finally:
            if real_env:
                real_env.send_action(np.zeros(6), 0)
                time.sleep(1)
                real_env.close()
            print("--- Real-World Phase Finished ---")
            
    # --- PHASE 3: Analyze and Save All Results ---
    if not all_results:
        print("\nNo data was generated. Exiting analysis."); return
        
    print("\n--- PHASE 3: Analyzing and Saving Results ---")
    results_df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    results_df.to_csv(csv_path, index=False)
    print(f"\nFull evaluation data saved to {csv_path}")

    # Generate the combined Sim2Real plots
    create_success_scatter_plot(results_df, os.path.join(OUTPUT_DIR, "sim2real_scatter.png"))
    create_detailed_latex_table(results_df, os.path.join(OUTPUT_DIR, LATEX_TABLE_FILENAME))
    
    print("\n--- Generating Real-World Performance Plots ---")
    real_perf_base_path = os.path.join(OUTPUT_DIR, "real_env_performance")
    
    # --- THIS IS THE FIX ---
    # We no longer need to merge. We just need to filter the complete DataFrame.
    # The 'initial_dist' is already correctly associated with each trial.
    real_world_df = results_df[results_df['environment'] == 'Real World'].copy()
    
    if not real_world_df.empty:
        plot_real_success_rate(real_world_df, real_perf_base_path)
        plot_real_relative_distance(real_world_df, real_perf_base_path)
        plot_real_absolute_distance(real_world_df, real_perf_base_path)
    else:
        print("No real-world data available to generate performance plots.")
    
    print("\n--- Evaluation Fully Complete ---")

if __name__ == '__main__':
    main()