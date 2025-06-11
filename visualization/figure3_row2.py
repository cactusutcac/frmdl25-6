import pandas as pd
import matplotlib.pyplot as plt

def plot_quantization_curve(csv_path, dataset_name, output_file):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 6))
    plt.step(df["input"], df["init_output"], where="post",
         label="Initialized", color="blue", linestyle="--", alpha=0.5)
    plt.step(df["input"], df["learned_output"], where="post",
         label="Learned", color="orange", linestyle="-", alpha=0.5)
    plt.xlabel("Input", fontsize=20)
    plt.ylabel("Output", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

plot_quantization_curve(
    "visualization/logs/quant_steps_kth.csv", 
    "KTH",
    "visualization/pics/learned_steps_kth.pdf"
)

plot_quantization_curve(
    "visualization/logs/quant_steps_ixmas.csv", 
    "IXMAS",
    "visualization/pics/learned_steps_ixmas.pdf"
)
