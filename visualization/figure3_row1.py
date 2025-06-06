import pandas as pd
import matplotlib.pyplot as plt

def plot_tradeoff(csv_bdq, csv_orig, dataset_name, output_file):
    # Load CSVs
    df_bdq = pd.read_csv(csv_bdq)
    df_orig = pd.read_csv(csv_orig)

    # Extract last epoch accuracies
    action_bdq = df_bdq['action_accuracy'].iloc[-1]
    privacy_bdq = df_bdq['privacy_accuracy'].iloc[-1]
    action_orig = df_orig['action_accuracy'].iloc[-1]
    privacy_orig = df_orig['privacy_accuracy'].iloc[-1]

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(privacy_orig, action_orig, label="Orig. Video", marker='o', s=100)
    plt.scatter(privacy_bdq, action_bdq, label="BDQ", marker='*', s=100)
    plt.plot([0, 100], [0, 100], 'k--', label='Ideal')

    plt.xlabel("Identity Accuracy (%)")
    plt.ylabel("Action Accuracy (%)")
    plt.title(f"{dataset_name}: Performance Trade-Off")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# KTH
plot_tradeoff(
    "accuracy_log_kth.csv",
    "accuracy_log_kth_no_encoder.csv",
    "KTH",
    "visualization/kth_tradeoff.png"
)

# IXMAS
plot_tradeoff(
    "accuracy_log_ixmas.csv",
    "accuracy_log_ixmas_no_encoder.csv",
    "IXMAS",
    "visualization/ixmas_tradeoff.png"
)
