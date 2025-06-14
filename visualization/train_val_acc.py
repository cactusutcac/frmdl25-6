import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_dual_accuracy_curve(log_root, output_path, title):
    tags = {
        "Train Action": "Accuracy_train_acc_action",
        "Val Action": "Accuracy_val_acc_action",
        "Train Privacy": "Accuracy_train_acc_privacy",
        "Val Privacy": "Accuracy_val_acc_privacy"
    }

    plt.figure(figsize=(12, 6))
    for label, subdir in tags.items():
        path = os.path.join(log_root, subdir)
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
        plt.plot(steps, scores, label=label)

    # plt.title(title)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

plot_dual_accuracy_curve(
    log_root="visualization/logs/runs_kth",
    output_path="visualization/pics/kth_accuracy_curve.pdf",
    title="KTH: Action vs. Privacy Accuracy Over Time"
)

plot_dual_accuracy_curve(
    log_root="visualization/logs/runs_ixmas",
    output_path="visualization/pics/ixmas_accuracy_curve.pdf",
    title="IXMAS: Action vs. Privacy Accuracy Over Time"
)
