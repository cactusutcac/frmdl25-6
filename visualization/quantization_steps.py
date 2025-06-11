import torch
import pandas as pd

# Row 2 Fig. 3
def save_quantizer_mapping(dq_module, output_csv_path="quant_steps.csv", device="cpu"):
    # Parameters
    num_bins = dq_module.num_bins
    hardness = dq_module.hardness
    normalize = dq_module.normalize_input
    rescale = dq_module.rescale_output

    # Input values in normalized space: [0, num_bins]
    x_vals = torch.linspace(0, num_bins, 1000, device=device).view(-1, 1)

    # Initial bin centers [0.5, 1.5, ..., 14.5]
    init_bins = torch.linspace(0.5, num_bins - 0.5, steps=num_bins).to(device).view(1, -1)

    # Learned bins
    learned_bins = dq_module.bins.detach().to(device).view(1, -1)

    # Quantization function: sum of sigmoids
    def quant_output(x, bins):
        return torch.sigmoid(hardness * (x - bins)).sum(dim=-1)

    # Evaluate
    with torch.no_grad():
        y_init = quant_output(x_vals, init_bins)
        y_learned = quant_output(x_vals, learned_bins)

        # Optional: rescale output like your quantizer does
        if rescale:
            y_init = y_init * (1.0) + 0.0  # No orig_min/max: we stay in normalized space
            y_learned = y_learned * (1.0) + 0.0

        # Convert to numpy
        x_vals_np = x_vals.squeeze().cpu().numpy()
        y_init_np = y_init.squeeze().cpu().numpy()
        y_learned_np = y_learned.squeeze().cpu().numpy()

    # Save as CSV
    df = pd.DataFrame({
        "input": x_vals_np,
        "init_output": y_init_np,
        "learned_output": y_learned_np
    })
    df.to_csv(output_csv_path, index=False)
