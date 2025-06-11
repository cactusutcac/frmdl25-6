import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_model_comparison(
    log_root_1, log_root_2,
    label_1="BDQ", label_2="Orig. Video",
    output_path="comparison.pdf", dataset_name="IXMAS"
):
    tags = {
        "Action Accuracy": "Accuracy_val_acc_action",
        "Privacy Accuracy": "Accuracy_val_acc_privacy"
    }

    plt.figure(figsize=(12, 8))
    for metric_name, tag_dir in tags.items():
        for root, label in [(log_root_1, label_1), (log_root_2, label_2)]:
            path = os.path.join(root, tag_dir)
            event_files = [f for f in os.listdir(path) if f.startswith("events.out")]
            if not event_files:
                print(f"No event file found in {path}")
                continue
            event_path = os.path.join(path, event_files[0])

            ea = event_accumulator.EventAccumulator(event_path)
            ea.Reload()

            if "Accuracy" not in ea.Tags()["scalars"]:
                print(f"Tag 'Accuracy' not found in {event_path}")
                continue

            values = ea.Scalars("Accuracy")
            steps = [v.step for v in values]
            scores = [v.value * 100 for v in values]
            plt.plot(steps, scores, label=f"{label} - {metric_name}")

    # plt.title(f"{dataset_name}: Validation Accuracy Comparison")
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

plot_model_comparison(
    log_root_1="visualization/logs/runs_kth",
    log_root_2="visualization/logs/runs_kth_no_encoder",
    label_1="BDQ",
    label_2="Orig. Video",
    output_path="visualization/pics/kth_model_comparison.pdf",
    dataset_name="KTH"
)

plot_model_comparison(
    log_root_1="visualization/logs/runs_ixmas",
    log_root_2="visualization/logs/runs_ixmas_no_encoder",
    label_1="BDQ",
    label_2="Orig. Video",
    output_path="visualization/pics/ixmas_model_comparison.pdf",
    dataset_name="IXMAS"
)
