import numpy as np


def save_model(model, filepath):
    state = {}
    parameter_index = 0
    layer_index = 0

    for layer in model.layers:
        if hasattr(layer, "parameters"):
            for parameter in layer.parameters():
                state[f"parameter_{parameter_index}"] = parameter.data.copy()
                parameter_index += 1

        if hasattr(layer, "running_mean") and hasattr(layer, "running_var"):
            state[f"layer_{layer_index}_running_mean"] = (
                layer.running_mean.copy()
            )
            state[f"layer_{layer_index}_running_var"] = (
                layer.running_var.copy()
            )

        layer_index += 1

    np.savez(filepath, **state)
    print(f"Model saved successfully to: {filepath}")


def load_model(model, filepath):
    with np.load(filepath, allow_pickle=False) as saved:
        parameter_index = 0

        for layer_index, layer in enumerate(model.layers):
            if hasattr(layer, "parameters"):
                for parameter in layer.parameters():
                    key = f"parameter_{parameter_index}"

                    if key not in saved:
                        raise ValueError(
                            f"Missing parameter in checkpoint: {key}"
                        )

                    saved_parameter = saved[key]

                    if saved_parameter.shape != parameter.data.shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: "
                            f"checkpoint {saved_parameter.shape}, "
                            f"model {parameter.data.shape}"
                        )

                    parameter.data[...] = saved_parameter
                    parameter_index += 1

            if hasattr(layer, "running_mean") and hasattr(layer, "running_var"):
                mean_key = f"layer_{layer_index}_running_mean"
                var_key = f"layer_{layer_index}_running_var"

                if mean_key not in saved or var_key not in saved:
                    raise ValueError(
                        "Checkpoint is missing Batch Normalization "
                        "running statistics."
                    )

                saved_mean = saved[mean_key]
                saved_var = saved[var_key]

                if saved_mean.shape != layer.running_mean.shape:
                    raise ValueError(
                        f"Running mean shape mismatch at layer {layer_index}."
                    )

                if saved_var.shape != layer.running_var.shape:
                    raise ValueError(
                        f"Running variance shape mismatch at layer {layer_index}."
                    )

                layer.running_mean[...] = saved_mean
                layer.running_var[...] = saved_var

    print(f"Model loaded successfully from: {filepath}")