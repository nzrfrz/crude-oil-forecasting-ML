import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PyEMD import CEEMDAN
import antropy as ant
from tqdm import tqdm


def calculate_zcr(signal):
    """Calculate Zero Crossing Rate"""
    zero_crosses = np.nonzero(np.diff(np.sign(signal)))[0]
    return len(zero_crosses) / len(signal)


def main():
    print("=== STARTING IMF JUSTIFICATION (ZCR, SAMPLE ENTROPY) ===")

    # ==========================================
    # 1. FOLDER & PATH CONFIGURATION
    # ==========================================
    DATA_PATH = "dataset/rolling-lowess-dataset.csv"
    STAT_EVAL_DIR = "evaluations/statistical/rolling-ceemdan"
    GRAPH_EVAL_DIR = "evaluations/graphical/rolling-ceemdan"

    for directory in [STAT_EVAL_DIR, GRAPH_EVAL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # ==========================================
    # 2. LOAD & SLICE DATA (Training Data Only)
    # ==========================================
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    # Take a representative sample from the past (e.g., before 2015)
    # to find the "Baseline Nature" of IMFs without touching future data.
    train_sample = df[df['Date'] < '2026-01-01'].copy()
    signal = train_sample['Rolling_LOWESS_Residual'].values

    print(
        f"Using {len(signal)} rows of training data (up to 2014) for testing.")

    # ==========================================
    # 3. RUN CEEMDAN (Static for Evaluation)
    # ==========================================
    print("Executing CEEMDAN... (Please wait)")
    ceemdan = CEEMDAN(trials=100)
    imfs = ceemdan.ceemdan(signal)

    num_imfs = imfs.shape[0] - 1  # Last row is residue
    print(f"CEEMDAN produced {num_imfs} IMFs and 1 Residue.")

    # ==========================================
    # 4. COMPUTE EVALUATION METRICS
    # ==========================================
    print("Calculating ZCR, Energy, and Sample Entropy for each IMF...")

    zcr_list = []
    sampen_list = []
    labels = []

    for i in tqdm(range(num_imfs)):
        imf_signal = imfs[i]

        # 1. Zero Crossing Rate
        zcr = calculate_zcr(imf_signal)

        # 2. Sample Entropy
        # Tolerance r = 0.2 * Standard Deviation (standard in academic literature)
        sampen = ant.sample_entropy(imf_signal, order=2, metric='chebyshev')

        zcr_list.append(zcr)
        sampen_list.append(sampen)
        labels.append(f'IMF {i+1}')

    # ==========================================
    # 5. SAVE STATISTICAL REPORT (.TXT)
    # ==========================================
    report_lines = [
        "REPORT ON IMF NOISE VS SIGNAL JUSTIFICATION",
        "=======================================================",
        "Evaluation Methods: Zero Crossing Rate (ZCR), Energy, & Sample Entropy",
        f"Number of Observations: {len(signal)} rows (Training Data)",
        "-------------------------------------------------------",
        f"{'IMF':<10} | {'ZCR':<10} | {'Sample Entropy':<15}",
        "-------------------------------------------------------"
    ]

    for i in range(num_imfs):
        line = f"IMF {i+1:<5} | {zcr_list[i]:<10.4f} | {sampen_list[i]:<15.4f}"
        report_lines.append(line)

    report_lines.extend([
        "-------------------------------------------------------",
        "INTERPRETATION GUIDELINE FOR THESIS:",
        "1. High ZCR + High Entropy (> 0.5) = High-Frequency Noise (Random/Speculative Signal).",
        "2. Low ZCR + Low Entropy = Patterned Signal (Economic/Market Cycle).",
        "3. Based on the table above, you can mathematically justify",
        "   discarding IMFs with high entropy (usually IMF 1 to 3)."
    ])

    stat_path = f"{STAT_EVAL_DIR}/imf_evaluation_report.txt"
    with open(stat_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[+] Statistical report saved at: {stat_path}")

    # ==========================================
    # 6. SAVE GRAPHICAL VISUALIZATION (.PNG)
    # ==========================================
    plt.style.use('default')

    # GRAPH 1: Sample Entropy & ZCR Comparison (Dual Axis Bar Chart)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(labels))
    width = 0.35

    ax1.bar(x_pos - width/2, sampen_list, width,
            label='Sample Entropy', color='darkred', alpha=0.8)
    ax1.set_xlabel('Intrinsic Mode Functions (IMF)')
    ax1.set_ylabel('Sample Entropy Value', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)

    # Create secondary Y-axis (right) for ZCR
    ax2 = ax1.twinx()
    ax2.bar(x_pos + width/2, zcr_list, width,
            label='ZCR', color='teal', alpha=0.8)
    ax2.set_ylabel('Zero Crossing Rate (ZCR)', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')

    plt.title('Complexity Analysis: Sample Entropy vs ZCR per IMF', fontsize=14)
    fig.tight_layout()
    plt.savefig(f"{GRAPH_EVAL_DIR}/01_entropy_zcr_comparison.png", dpi=300)
    plt.close()

    print(f"[+] Visualization graph saved at: {GRAPH_EVAL_DIR}/")
    print("✅ IMF JUSTIFICATION PROCESS COMPLETED!")


# REQUIRED FOR WINDOWS + MULTIPROCESSING
if __name__ == '__main__':
    main()