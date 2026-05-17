"""
rolling-ceemdan.py
Rolling CEEMDAN decomposition on LOWESS residuals.
Extracts IMFs (1..10) + residue, reconstructs signal using IMF 4+.
Zero leakage (rolling window uses only past data).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PyEMD import CEEMDAN
from tqdm import tqdm


def plot_global_imfs_viz(df_clean, output_dir):
    """
    Generates stacked IMF plot (full history) using different colors
    for noise vs signal spectrum.
    """
    print("\n[V] Creating Global IMF Spectrum Visualization (Full History)...")

    viz_signal = df_clean['Rolling_LOWESS_Residual'].values
    viz_dates = df_clean['Date'].values

    # Use global CEEMDAN (full series, not rolling)
    viz_ceemdan = CEEMDAN(trials=50)
    print("   -> Computing Global CEEMDAN (may take 1-3 minutes)...")
    viz_imfs = viz_ceemdan.ceemdan(viz_signal)

    num_imfs = viz_imfs.shape[0] - 1
    residue = viz_imfs[-1]

    total_rows = num_imfs + 1
    fig, axes = plt.subplots(total_rows, 1, figsize=(15, 2 * total_rows), sharex=True)

    noise_cutoff_idx = 2  # IMF 1, 2, 3 are considered noise

    for i in range(num_imfs):
        ax = axes[i]
        imf_num = i + 1

        if i <= noise_cutoff_idx:
            color = 'salmon'
            label_suffix = " (Filtered Noise)"
        else:
            color = 'darkblue'
            label_suffix = " (Chosen Signal)"

        ax.plot(viz_dates, viz_imfs[i], color=color, linewidth=1)
        ax.set_ylabel(f'IMF {imf_num}{label_suffix}',
                      fontsize=12, rotation=0, labelpad=40, ha='right')
        ax.grid(True, linestyle='--', alpha=0.3)

    ax_res = axes[-1]
    ax_res.plot(viz_dates, residue, color='black', linewidth=1)
    ax_res.set_ylabel('CEEMDAN\nResidue', fontsize=12,
                      rotation=0, labelpad=40, ha='right')
    ax_res.grid(True, linestyle='--', alpha=0.3)
    ax_res.set_xlabel('Date', fontsize=12)

    fig.suptitle('Global CEEMDAN Decomposition Spectrum (For Visualization)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    global_plot_path = f'{output_dir}/02_global_ceemdan_components.png'
    plt.savefig(global_plot_path, dpi=300)
    plt.close()
    print(f"[+] Global IMF spectrum plot saved at: {global_plot_path}")


def main():
    print("=== STARTING ROLLING CEEMDAN PIPELINE (0% LEAKAGE) ===")

    # ==========================================
    # 1. FOLDER & PARAMETER CONFIGURATION
    # ==========================================
    DATASET_DIR = "dataset"
    STAT_EVAL_DIR = "evaluations/statistical/rolling-ceemdan"
    GRAPH_EVAL_DIR = "evaluations/graphical/rolling-ceemdan"

    for directory in [STAT_EVAL_DIR, GRAPH_EVAL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    window_size = 252
    MAX_IMFS = 10

    # ==========================================
    # 2. LOAD LOWESS OUTPUT DATASET
    # ==========================================
    data_path = f'{DATASET_DIR}/rolling-lowess-dataset.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    target_signal = df['Rolling_LOWESS_Residual'].values

    # ==========================================
    # 3. PREPARE ARRAYS FOR ML FEATURES
    # ==========================================
    imf_results = {f'IMF_{i}': np.full(len(target_signal), np.nan) for i in range(1, MAX_IMFS + 1)}
    ceemdan_residue = np.full(len(target_signal), np.nan)
    reconstructed_signal = np.full(len(target_signal), np.nan)

    # Memory/CPU optimization: 50 trials, limit processes to limit_cores
    limit_cores = 4
    rolling_trials = 50
    rolling_ceemdan = CEEMDAN(trials=rolling_trials, processes=limit_cores)

    # ==========================================
    # 4. ROLLING CEEMDAN PROCESS (WITH TIMER)
    # ==========================================
    print(f"Processing {len(target_signal)} data rows with rolling window...")
    print(f"Using {rolling_trials} trials and {limit_cores} CPU cores.")

    start_time = time.time()

    for i in tqdm(range(window_size, len(target_signal)), desc="Rolling CEEMDAN"):
        window_data = target_signal[i - window_size + 1 : i + 1]
        imfs = rolling_ceemdan.ceemdan(window_data)

        current_imf_values = imfs[:, -1]
        num_imfs_extracted = imfs.shape[0] - 1

        # Store IMF values (or zero if fewer than MAX_IMFS)
        for j in range(num_imfs_extracted):
            if j < MAX_IMFS:
                imf_results[f'IMF_{j+1}'][i] = current_imf_values[j]

        for j in range(num_imfs_extracted, MAX_IMFS):
            imf_results[f'IMF_{j+1}'][i] = 0.0

        ceemdan_residue[i] = current_imf_values[-1]

        # Reconstruction: sum IMF 4+ (index 3 onward)
        if num_imfs_extracted >= 3:
            reconstructed_signal[i] = np.sum(current_imf_values[3:])
        else:
            reconstructed_signal[i] = current_imf_values[-1]

    end_time = time.time()
    execution_time_seconds = end_time - start_time

    # Format execution time
    hours, rem = divmod(execution_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    time_formatted = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    # ==========================================
    # 5. MERGE RESULTS & REMOVE NaN ROWS
    # ==========================================
    for j in range(1, MAX_IMFS + 1):
        df[f'IMF_{j}'] = imf_results[f'IMF_{j}']

    df['CEEMDAN_Residue'] = ceemdan_residue
    df['Reconstructed_Signal'] = reconstructed_signal
    df_clean = df.dropna(subset=['Reconstructed_Signal']).reset_index(drop=True)

    # ==========================================
    # 6. SAVE MASTER DATASET
    # ==========================================
    output_csv = f'{DATASET_DIR}/master-features-dataset.csv'
    df_clean.to_csv(output_csv, index=False)

    print(f"\n[+] Master dataset (ready for ML) saved at: {output_csv}")

    # ==========================================
    # 7. GRAPHICAL EVALUATION
    # ==========================================
    print("\nGenerating reconstruction visualisation...")
    plt.style.use('default')
    plot_data = df_clean.tail(500)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_data['Date'], plot_data['Rolling_LOWESS_Residual'],
             label='Raw Noisy Residual', color='lightgray', alpha=0.8)
    plt.plot(plot_data['Date'], plot_data['Reconstructed_Signal'],
             label='Reconstructed Signal (IMF 4+)', color='darkblue', linewidth=2)
    plt.title('Denoised Signal Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{GRAPH_EVAL_DIR}/01_reconstruction_comparison.png', dpi=300)
    plt.close()

    plot_global_imfs_viz(df_clean, GRAPH_EVAL_DIR)

    # ==========================================
    # 8. STATISTICAL EVALUATION (with execution time)
    # ==========================================
    report_text = f"""
        "ROLLING CEEMDAN & RECONSTRUCTION REPORT\n"
        "===========================================\n"
        f"Total Input Data      : {len(target_signal)} rows\n"
        f"CEEMDAN Window Size   : {window_size} days\n"
        f"Trials (per window)   : {rolling_trials}\n"
        f"Total Master Data (ML): {len(df_clean)} rows\n"
        f"Total Computation Time: {time_formatted}\n\n"
        "RECONSTRUCTION METHOD:\n"
        "Based on Sample Entropy and Zero Crossing Rate (ZCR) tests,\n"
        "IMFs 1, 2, and 3 are identified as high‑frequency noise.\n"
        "The 'Reconstructed_Signal' is formed by summing IMF 4\n"
        "through the last IMF plus the CEEMDAN residue.\n"
        "===========================================\n"
    """

    stat_output = f'{STAT_EVAL_DIR}/ceemdan_reconstruction_report.txt'
    with open(stat_output, 'w', encoding='utf-8') as text_file:
        text_file.write(report_text)

    print(f"\n[+] Report and graphs saved in evaluations/")
    print(f"⏳ EXECUTION TIME: {time_formatted}")
    print("✅ ROLLING CEEMDAN PIPELINE COMPLETED!")


if __name__ == '__main__':
    main()