import pandas as pd
import matplotlib.pyplot as plt

def plot_quantization_curve(csv_path, dataset_name, output_file):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 6))
    plt.step(df["input"], df["init_output"], where="post", label="Initialized", color="blue")
    plt.step(df["input"], df["learned_output"], where="post", label="Learned", color="orange")
    plt.xlabel("Input")
    plt.ylabel("Output")
    # plt.title(f"Quantization Mapping ({dataset_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

plot_quantization_curve(
    "visualization/logs/quant_steps_kth.csv", 
    "KTH"
    "visualization/pics/learned_steps_kth.pdf"
)

plot_quantization_curve(
    "visualization/logs/quant_steps_ixmas.csv", 
    "IXMAS"
    "visualization/pics/learned_steps_ixmas.pdf"
)
