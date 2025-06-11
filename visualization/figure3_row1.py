import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_last_scalar_from_subdir(root_dir, subdir_name):
    """
    Extracts the last scalar value from a TensorBoard run directory. 
    Assumes:
    - Subdirectory is named by the metric (e.g., Accuracy_val_acc_action)
    - Scalar tag inside is always 'Accuracy'
    """
    full_path = os.path.join(root_dir, subdir_name)
    event_files = [f for f in os.listdir(full_path) if f.startswith("events.out")]
    if not event_files:
        raise FileNotFoundError(f"No event file found in {full_path}")

    event_path = os.path.join(full_path, event_files[0])
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    if "Accuracy" not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag 'Accuracy' not found in {event_path}")

    scalars = ea.Scalars("Accuracy")
    return scalars[-1].value

def plot_tradeoff_tensorboard(log_root_bdq, log_root_orig, tag_action, tag_privacy, dataset_name, output_file):
    action_bdq = extract_last_scalar_from_subdir(log_root_bdq, tag_action)
    privacy_bdq = extract_last_scalar_from_subdir(log_root_bdq, tag_privacy)
    action_orig = extract_last_scalar_from_subdir(log_root_orig, tag_action)
    privacy_orig = extract_last_scalar_from_subdir(log_root_orig, tag_privacy)

    # Convert to %
    action_bdq *= 100
    privacy_bdq *= 100
    action_orig *= 100
    privacy_orig *= 100

    plt.figure(figsize=(8,8))
    plt.scatter(privacy_orig, action_orig, label="Orig. Video", marker='o', s=100)
    plt.scatter(privacy_bdq, action_bdq, label="BDQ", marker='*', s=100)
    plt.xlabel("Identity Accuracy (%)")
    plt.ylabel("Action Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

plot_tradeoff_tensorboard(
    log_root_bdq="visualization/logs/runs_ixmas",
    log_root_orig="visualization/logs/runs_ixmas_no_encoder",
    tag_action="Accuracy_val_acc_action",
    tag_privacy="Accuracy_val_acc_privacy",
    dataset_name="IXMAS",
    output_file="visualization/pics/tradeoff_ixmas.pdf"
)
