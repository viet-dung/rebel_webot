import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Using seaborn for prettier, easier boxplots
from stable_baselines3 import SAC
from tqdm import tqdm
import time

# Import the environment class from your other file
from realEnv import RealEnv 
from simulationEnv import SimulationEnv

USE_LATEX = True 

# These settings are used only if USE_LATEX is True
if USE_LATEX:
    from matplotlib import rcParams
    
    # To get your document's text width, use `\the\textwidth` in your LaTeX file
    textwidth_pt = 427.43153
    pt_to_inch = 1.0 / 72.27
    textwidth_in = textwidth_pt * pt_to_inch
    
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0
    fig_width_in = textwidth_in * 1.5
    fig_height_in = fig_width_in * golden_ratio * 0.6

    rcParams.update({
        "font.family": "serif", 
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    print("INFO: LaTeX rendering is ENABLED.")
else:
    # Fallback settings for standard Matplotlib rendering
    fig_width_in = 20
    fig_height_in = 7
    print("INFO: Standard Matplotlib rendering is enabled (USE_LATEX=False).")

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
NUM_EVAL_EPISODES = 100 # Set to 1000 for your full run, 100 is good for testing
SUCCESS_THRESHOLD = 0.01 # Distance in meters to be considered a success

POLICY_ORDER = ['SAC_23', 'SAC_19', 'Random']
POLICY_PALETTE = {
    'SAC_23': 'dodgerblue',      # Color for your first model
    'SAC_19': 'mediumseagreen',  # A new color for your second model
    'Random': 'salmon'           # Color for the random policy
}

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
model1_path = os.path.join(CODE_DIR, "sac_stack_best", "SAC_reaching_4130000_steps.zip") 
model2_path = os.path.join(CODE_DIR, "sac_stack_best", "SAC_reaching_3430000_steps.zip")

OUTPUT_DIR = "plots/sim"
OUTPUT_CSV_FILE = "evaluation_results_focused.csv"
OUTPUT_PLOT_FILE = "evaluation_.png"
ENV_PORT = '1244'
ENV_ID = 10

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_plots(df, filename):
    """
    Generates a 1x3 plot using the pre-configured document style, saving as a PNG.
    """
    # Use the globally defined figure size from the configuration block
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width_in, fig_height_in))
    fig.suptitle('Focused Policy Performance Evaluation')
    palette = POLICY_PALETTE
    output_filename = os.path.splitext(filename)[0] + ".png"

    # --- Plot 1: Success Rate (Bar Plot) ---
    ax = axes[0]
    success_rates = df.groupby('policy')['is_success'].mean() * 100
    success_df = success_rates.reset_index().rename(columns={'is_success': 'rate'})
    
    sns.barplot(ax=ax, x='policy', y='rate', data=success_df, hue='policy', palette=palette,order=POLICY_ORDER, legend=False)
    
    ax.set_title('Success Rate')
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Policy')
    ax.set_ylim(0, 105)
    for i, rate in enumerate(success_rates.reindex(['SAC', 'Random'])):
        ax.text(i, rate - 10 if rate > 50 else rate + 2, f'{rate:.1f}%', ha='center')

    # --- Plot 2: Relative Final Distance to Target (Box Plot) ---
    ax = axes[1]
    sns.boxplot(ax=ax, x='policy', y='final_dist', data=df, hue='policy', palette=palette,order=POLICY_ORDER, legend=False)
    ax.axhline(SUCCESS_THRESHOLD, color='green', linestyle='--', linewidth=1.5, label=f'Success Threshold\n({SUCCESS_THRESHOLD}m)')
    ax.set_yscale('log')
    ax.set_title('Distribution of Final Distances')
    ax.set_ylabel('Final Distance (meters, log scale)')
    ax.set_xlabel('Policy')
    ax.legend()
    
    # --- Plot 3: Steps to Success (Box Plot) ---
    ax = axes[2]
    successful_df = df[df['is_success']].copy()
    
    if not successful_df.empty:
        policies_with_success = successful_df['policy'].unique()
        order = [p for p in ['SAC', 'Random'] if p in policies_with_success]
        sns.boxplot(ax=ax, x='policy', y='steps_taken', data=successful_df, hue='policy', palette=palette, order=order, legend=False)
        ax.set_title('Steps to Success')
        ax.set_ylabel('Number of Environment Steps')
        ax.set_xlabel('Policy')
    else:
        ax.text(0.5, 0.5, "No successful episodes to plot.", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Steps to Success')
        ax.set_xticks([])
        ax.set_yticks([])

    # Use tight_layout to automatically adjust spacing
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to {output_filename}")
    plt.show()

def create_success_rate_plot(df, filename):
    """
    Generates a single, clear bar chart for the success rate.
    """
    # Use the globally defined figure size, but make it less wide for a single plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width_in / 2.5, fig_height_in * 1.2))
    
    palette = POLICY_PALETTE
    output_filename = os.path.splitext(filename)[0] + "_success_rate.pdf"

    success_rates = df.groupby('policy')['is_success'].mean() * 100
    success_df = success_rates.reset_index().rename(columns={'is_success': 'rate'})
    
    sns.barplot(ax=ax, x='policy', y='rate', data=success_df, hue='policy', palette=palette,order=POLICY_ORDER, legend=False)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Policy')
    ax.set_ylim(0, 105)
    
    for i, rate in enumerate(success_rates.reindex(POLICY_PALETTE)):

        text_y, color, va = rate + 2, 'black', 'bottom'
        ax.text(i, text_y, f'{rate:.1f}%', ha='center', color=color, va=va, fontweight='bold')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"\nSuccess rate plot saved to {output_filename}")


def create_relative_distance_plot(df, filename):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width_in / 2.5, fig_height_in * 1.2))
    palette = POLICY_PALETTE
    output_filename = os.path.splitext(filename)[0] + "_relative.pdf"

    # Calculate progress percentage
    df['progress_perc'] = 100 * (df['initial_dist'] - df['final_dist']) / df['initial_dist'].replace(0, 1e-6)
    
    sns.boxplot(ax=ax, x='policy', y='progress_perc', data=df, hue='policy', palette=palette,order=POLICY_ORDER, legend=False)
    sns.stripplot(ax=ax, x='policy', y='progress_perc', data=df, hue='policy', palette=palette, order=POLICY_ORDER, legend=False, jitter=True, s=8)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='100% Progress')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='0% Progress')
    ax.set_ylabel('Progress (%)')
    ax.set_xlabel('Policy')
    ax.set_ylim(-50, 150)
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, bbox_inches="tight")

def create_absolute_distance_plot(df, filename):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width_in / 2.5, fig_height_in * 1.2))
    palette = POLICY_PALETTE
    output_filename = os.path.splitext(filename)[0] + "_absolute.pdf"

    sns.boxplot(ax=ax, x='policy', y='final_dist', data=df, hue='policy', palette=palette,order=POLICY_ORDER, legend=False)
    sns.stripplot(ax=ax, x='policy', y='final_dist', data=df, hue='policy', palette=palette, order=POLICY_ORDER, legend=False, jitter=True, s=8)
    ax.axhline(SUCCESS_THRESHOLD, color='green', linestyle='--', linewidth=1.5, label=f'Success Threshold\n({SUCCESS_THRESHOLD}m)')
    ax.set_yscale('log')
    ax.set_ylabel('Final Distance (meters, log scale)')
    ax.set_xlabel('Policy')
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, bbox_inches="tight")

def create_step_plot(df, filename):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width_in / 2.5, fig_height_in * 1.2))
    palette = POLICY_PALETTE
    output_filename = os.path.splitext(filename)[0] + "_step.pdf"

    # Calculate progress percentage
    successful_df = df[df['is_success']].copy()
    if not successful_df.empty:
        policies_with_success = successful_df['policy'].unique()
        order = [p for p in POLICY_ORDER if p in policies_with_success]
        sns.boxplot(ax=ax, x='policy', y='steps_taken', data=successful_df, hue='policy', palette=palette, order=order, legend=False)
        ax.set_ylabel('Number of Steps')
        ax.set_xlabel('Policy')
    else:
        ax.text(0.5, 0.5, "No successful episodes to plot.", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Efficiency: Steps to Success')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, bbox_inches="tight")

def save_latex_summary_table(df, output_path):
    """
    Calculates and saves a publication-quality LaTeX-formatted summary table.
    This final version handles NaN from single-episode evaluations and fixes
    the multicolumn alignment bug.
    """
    # --- 1. Calculate stats, using .fillna(0) to handle std of a single sample ---
    success_rates = df.groupby('policy')['is_success'].mean() * 100
    
    df_copy = df.copy()
    df_copy['progress_perc'] = 100 * (df_copy['initial_dist'] - df_copy['final_dist']) / df_copy['initial_dist'].replace(0, 1e-6)
    
    # === FIX for NaN: Add .fillna(0) after calculating stats ===
    progress_stats = df_copy.groupby('policy')['progress_perc'].agg(['mean', 'std']).fillna(0)
    dist_stats = df.groupby('policy')['final_dist'].agg(['mean', 'std']).fillna(0)
    
    successful_df = df[df['is_success']]
    if not successful_df.empty:
        # Also apply fillna(0) here for the case of one successful episode
        steps_stats = successful_df.groupby('policy')['steps_taken'].agg(['mean', 'std']).fillna(0)
    else:
        steps_stats = pd.DataFrame(
            {'mean': [np.nan, np.nan], 'std': [np.nan, np.nan]},
            index=pd.Index(['Random', 'SAC'], name='policy')
        )

    # --- 2. Build the DataFrame ---
    summary_df = pd.DataFrame(index=POLICY_ORDER)
    summary_df[r'Success (%)'] = success_rates.map('{:.1f}'.format)
    summary_df['Progress Mean'] = progress_stats['mean'].map('{:.1f}'.format)
    summary_df['Progress Std'] = progress_stats['std'].map('{:.1f}'.format)
    summary_df['Distance Mean'] = dist_stats['mean'].map('{:.3f}'.format)
    summary_df['Distance Std'] = dist_stats['std'].map('{:.3f}'.format)
    summary_df['Steps Mean'] = steps_stats['mean'].map('{:.1f}'.format)
    summary_df['Steps Std'] = steps_stats['std'].map('{:.1f}'.format)
    
    summary_df.loc['Random', 'Steps Mean'] = r'\multicolumn{2}{c}{--}'
    summary_df.loc['Random', 'Steps Std'] = '' # This is part of the problem, we will fix its output

    # --- 3. Generate the initial LaTeX string ---
    summary_df = summary_df.reset_index().rename(columns={'index': 'Policy'})
    header = [
        ('Policy', ''), ('Success (%)', ''),
        ('Progress (%)', 'Mean'), ('Progress (%)', 'Std.'),
        ('Final Distance (m)', 'Mean'), ('Final Distance (m)', 'Std.'),
        ('Steps to Success', 'Mean'), ('Steps to Success', 'Std.')
    ]
    summary_df.columns = pd.MultiIndex.from_tuples(header)

    latex_string = summary_df.to_latex(
        column_format='l c cc cc cc', index=False, escape=False, na_rep='--',
        caption='Mean and standard deviation of key performance metrics. `Steps to Success` is calculated only on successful episodes.',
        label='tab:performance_summary', position='!htbp'
    )
    
    # --- 4. Manually perfect the generated LaTeX string ---
    lines = latex_string.splitlines()
    
    # Fix Headers
    toprule_index = lines.index(r'\toprule')
    header_row1 = r'\textbf{Policy} & \textbf{Success (\%)} & \multicolumn{2}{c}{\textbf{Progress (\%)}} & \multicolumn{2}{c}{\textbf{Final Distance (m)}} & \multicolumn{2}{c}{\textbf{Steps to Success}} \\'
    header_row2 = r'               &                       & Mean & Std. & Mean & Std. & Mean & Std. \\'
    cmidrule_row = r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}'
    lines[toprule_index + 1] = header_row1
    lines.insert(toprule_index + 2, cmidrule_row)
    lines[toprule_index + 3] = header_row2
    
    # === FIX for extra '&': Find and replace the broken line ===
    for i, line in enumerate(lines):
        if r'\multicolumn{2}{c}{--} & ' in line:
            # Replace the sequence of "multicolumn & empty_cell" with just "multicolumn"
            lines[i] = line.replace(r'\multicolumn{2}{c}{--} & ', r'\multicolumn{2}{c}{--}')

    final_latex_string = '\n'.join(lines)

    # --- 5. Save the final LaTeX string ---
    table_filename = os.path.join(output_path, "summary_table.tex")
    with open(table_filename, 'w') as f:
        f.write(r'%% This table was automatically generated by eval.py %%' + '\n')
        f.write(final_latex_string)
    
    print(f"Publication-quality LaTeX summary table saved to: {table_filename}")

def evaluate_policy(env, policy_name, model=None, num_episodes=10, box_poses_to_use=None):
    all_episode_data = []
    generated_box_poses = []
    is_forced_pose_run = box_poses_to_use is not None

    for i in tqdm(range(num_episodes), desc=f"Evaluating {policy_name} Policy"):
        obs, _ = env.reset()
        if is_forced_pose_run:
            env.current_box_pose = np.array(box_poses_to_use[i])
            env.set_box_pose = False
            obs = env.get_observation()
        
        generated_box_poses.append(env.current_box_pose.copy())
        
        initial_gripper_pos = obs['achieved_goal'][0:3].copy()
        initial_box_pos = obs['achieved_goal'][7:10].copy()
        initial_dist = np.linalg.norm(initial_box_pos - initial_gripper_pos)

        terminated, truncated = False, False
        steps_taken = 0
        while not terminated and not truncated:
            action, _ = (model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None))
            obs, _, terminated, truncated, _ = env.step(action)
            steps_taken += 1
        
        is_success = terminated
        final_gripper_pos = obs['achieved_goal'][0:3].copy()
        final_box_pos = obs['achieved_goal'][7:10].copy()
        final_dist = np.linalg.norm(final_box_pos - final_gripper_pos)
        
        all_episode_data.append({
            'policy': policy_name,
            'initial_dist': initial_dist, # Add initial distance to results
            'final_dist': final_dist,
            'is_success': is_success,
            'steps_taken': steps_taken,
        })
    return all_episode_data, generated_box_poses
# In eval.py, replace your create_plots function with this one:

def run_evaluation(model1_path, model2_path, num_episodes, output_csv, output_plot, port, env_id):
    eval_env = None
    try:
        output_path = os.path.join(CODE_DIR, OUTPUT_DIR)
        os.makedirs(output_path, exist_ok=True)
        full_csv_path = os.path.join(output_path, output_csv)
        full_plot_path = os.path.join(output_path, output_plot)

        eval_env = SimulationEnv(port=port, env_id=env_id)
        
        print(f"Loading SAC model from {model1_path}...")
        sac_model = SAC.load(model1_path, env=eval_env)

        print(f"\n--- Running {num_episodes} evaluation episodes for each policy ---")
        
        print("\n--- Step 1: Evaluating SAC Policy ---")
        sac_results, _ = evaluate_policy(
            env=eval_env, policy_name="SAC", model=sac_model, num_episodes=1
        )

        sac_results, used_box_poses = evaluate_policy(
            env=eval_env, policy_name="SAC_23", model=sac_model, num_episodes=num_episodes
        )

        sac_model = SAC.load(model2_path, env=eval_env)
        sac19_results, _ = evaluate_policy(
            env=eval_env, policy_name="SAC_19", model=sac_model, num_episodes=num_episodes, box_poses_to_use=used_box_poses
        )

        print("\n--- Step 2: Evaluating Random Policy (on the same box locations) ---")
        random_results, _ = evaluate_policy(
            env=eval_env, policy_name="Random", model=None, num_episodes=num_episodes, box_poses_to_use=used_box_poses
        )

        results_df = pd.DataFrame(sac_results + sac19_results + random_results)
        results_df.to_csv(full_csv_path, index=False)
        print(f"\nFull evaluation data saved to {full_csv_path}")

        print("\n--- Summary Statistics ---")
        success_rate = results_df.groupby('policy')['is_success'].mean() * 100
        print("\nSuccess Rate (%):")
        print(success_rate)
        
        successful_df = results_df[results_df['is_success']]
        if not successful_df.empty:
            print("\nSteps to Success (for successful episodes):")
            print(successful_df.groupby('policy')['steps_taken'].describe())
        else:
            print("\nNo successful episodes were recorded for one or both policies.")

        save_latex_summary_table(results_df, output_path)
        create_success_rate_plot(results_df, full_plot_path)
        create_absolute_distance_plot(results_df, full_plot_path)
        create_relative_distance_plot(results_df, full_plot_path)
        create_step_plot(results_df, full_plot_path)
        

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
    finally:
        if eval_env:
            print("\nClosing environment...")
            eval_env.close()
        print("Evaluation finished.")

if __name__ == '__main__':
    run_evaluation(
        model1_path=model1_path,
        model2_path=model2_path,
        num_episodes=NUM_EVAL_EPISODES,
        output_csv=OUTPUT_CSV_FILE,
        output_plot=OUTPUT_PLOT_FILE,
        port=ENV_PORT,
        env_id=ENV_ID
    )