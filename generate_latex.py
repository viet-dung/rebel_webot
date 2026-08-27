import os
import pandas as pd
import numpy as np

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# --- PATHS AND FILENAMES ---
INPUT_DIR = "policy_comparison_results/policy_comparison_results"
INPUT_CSV_FILENAME = "sac_vs_random_raw_data.csv"

OUTPUT_DIR = "policy_comparison_results"
OUTPUT_LATEX_FILENAME = "performance_summary_table_multicolumn.tex"

# This is the new policy order we will use, derived from your data.
POLICY_ORDER = ['SAC (Simulation)', 'SAC (Real World)']

# ==============================================================================
# LATEX TABLE GENERATION FUNCTION (Adapted to be Generic)
# ==============================================================================

def save_latex_summary_table(df, policy_order, output_path, output_filename):
    """
    Calculates and saves a publication-quality LaTeX-formatted summary table
    with multi-column headers.
    
    This function is adapted from your provided version to be more general:
    - It accepts a `policy_order` list to define the rows.
    - It applies the special '--' multicolumn for 'Steps to Success' to *any*
      policy with zero successful runs, not just one hard-coded as 'Random'.
    """
    # --- 1. Calculate stats, reindexing to ensure all policies are present ---
    success_rates = df.groupby('policy')['is_success'].mean().reindex(policy_order).fillna(0) * 100
    
    df_copy = df.copy()
    df_copy['progress_perc'] = 100 * (df_copy['initial_dist'] - df_copy['final_dist']) / df_copy['initial_dist'].replace(0, 1e-6)
    
    progress_stats = df_copy.groupby('policy')['progress_perc'].agg(['mean', 'std']).reindex(policy_order).fillna(0)
    dist_stats = df.groupby('policy')['final_dist'].agg(['mean', 'std']).reindex(policy_order).fillna(0)
    
    successful_df = df[df['is_success']]
    if not successful_df.empty:
        # Reindex to ensure all policies are in the table, even if they have 0 successes (will be NaN)
        steps_stats = successful_df.groupby('policy')['steps_taken'].agg(['mean', 'std']).reindex(policy_order)
    else:
        # Create a DataFrame of NaNs if no policy had any success
        steps_stats = pd.DataFrame(
            {'mean': np.nan, 'std': np.nan},
            index=pd.Index(policy_order, name='policy')
        )

    # --- 2. Build the DataFrame ---
    summary_df = pd.DataFrame(index=policy_order)
    summary_df[r'Success (%)'] = success_rates.map('{:.1f}'.format)
    summary_df['Progress Mean'] = progress_stats['mean'].map('{:.1f}'.format)
    summary_df['Progress Std'] = progress_stats['std'].map('{:.1f}'.format)
    summary_df['Distance Mean'] = dist_stats['mean'].map('{:.3f}'.format)
    summary_df['Distance Std'] = dist_stats['std'].map('{:.3f}'.format)
    
    # Format steps, leaving NaNs for policies with no success
    summary_df['Steps Mean'] = steps_stats['mean'].map('{:.1f}'.format)
    summary_df['Steps Std'] = steps_stats['std'].map('{:.1f}'.format)
    
    # --- THIS IS THE GENERALIZED LOGIC ---
    # Instead of hard-coding for 'Random', we check for any policy that has
    # no successful runs (where steps_stats['mean'] would be NaN).
    policies_with_no_success = steps_stats[steps_stats['mean'].isna()].index
    for policy_name in policies_with_no_success:
        summary_df.loc[policy_name, 'Steps Mean'] = r'\multicolumn{2}{c}{--}'
        summary_df.loc[policy_name, 'Steps Std'] = '' # This makes the multicolumn span correctly

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
    
    # --- 4. Manually perfect the generated LaTeX string (Your logic, preserved) ---
    lines = latex_string.splitlines()
    
    toprule_index = lines.index(r'\toprule')
    header_row1 = r'\textbf{Policy} & \textbf{Success (\%)} & \multicolumn{2}{c}{\textbf{Progress (\%)}} & \multicolumn{2}{c}{\textbf{Final Distance (m)}} & \multicolumn{2}{c}{\textbf{Steps to Success}} \\'
    header_row2 = r'               &                       & Mean & Std. & Mean & Std. & Mean & Std. \\'
    cmidrule_row = r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}'
    lines[toprule_index + 1] = header_row1
    lines.insert(toprule_index + 2, cmidrule_row)
    lines[toprule_index + 3] = header_row2
    
    # Fix for the extra '&' created by pandas after a multicolumn cell
    for i, line in enumerate(lines):
        if r'\multicolumn{2}{c}{--} & ' in line:
            lines[i] = line.replace(r'\multicolumn{2}{c}{--} & ', r'\multicolumn{2}{c}{--}')

    final_latex_string = '\n'.join(lines)

    # --- 5. Save the final LaTeX string ---
    full_output_path = os.path.join(output_path, output_filename)
    os.makedirs(output_path, exist_ok=True)
    with open(full_output_path, 'w') as f:
        f.write(r'%% This table was automatically generated. Add \usepackage{booktabs} to your preamble. %%' + '\n\n')
        f.write(final_latex_string)
    
    print(f"✅ Publication-quality LaTeX summary table saved to: {full_output_path}")

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

def main():
    """
    Main function to read the CSV, process data, and generate the LaTeX table.
    """
    input_csv_path = os.path.join(INPUT_DIR, INPUT_CSV_FILENAME)
    if not os.path.exists(input_csv_path):
        print(f"❌ FATAL: Input CSV file not found at '{input_csv_path}'")
        return

    print(f"🔄 Reading data from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)

    print("🔄 Transforming data for comparison...")
    df['policy'] = df['policy'] + ' (' + df['environment'] + ')'
    
    print(f"📊 Found data for policies: {df['policy'].unique().tolist()}")

    # Call the adapted function with our data and desired policy order
    save_latex_summary_table(
        df=df,
        policy_order=POLICY_ORDER,
        output_path=OUTPUT_DIR,
        output_filename=OUTPUT_LATEX_FILENAME
    )
    
    print("\n--- Script finished successfully! ---")

if __name__ == '__main__':
    main()