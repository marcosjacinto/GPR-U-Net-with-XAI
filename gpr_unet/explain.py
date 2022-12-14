from pathlib import Path

import mlflow
import numpy as np
import shap
import tensorflow as tf

shap.explainers._deep.deep_tf.op_handlers[
    "Conv2DBackpropInput"
] = shap.explainers._deep.deep_tf.passthrough


def get_deep_explainer(
    run_id: str, background_samples: np.ndarray
) -> shap.DeepExplainer:

    run = mlflow.get_run(run_id)
    artifact_uri = Path(run.info.artifact_uri)
    model_path = artifact_uri / "model/data/model"

    model = tf.keras.models.load_model(model_path)
    flattened_model = tf.keras.models.Sequential()
    flattened_model.add(model)
    flattened_model.add(tf.keras.layers.Flatten())
    flattened_model.compile(loss="binary_crossentropy")

    return shap.DeepExplainer(flattened_model, background_samples)


def get_channel_contributions(shap_values: np.ndarray) -> np.ndarray:

    n_pixels, n_samples, n_img_size, _, n_channels = shap_values.shape

    swaped_shap_values = np.swapaxes(shap_values, 0, 1)
    
    channel_contributions = []

    for sample_n in range(n_samples):
    
        sample = swaped_shap_values[sample_n]
        summed_contributions = sample.sum(axis=1).sum(axis=1)
        channel_contributions.append(
            summed_contributions.reshape(n_img_size, n_img_size, n_channels)
            )

    channel_contributions = np.array(channel_contributions)

    return channel_contributions


def get_background_samples(
    x_data: np.ndarray, y_data: np.ndarray, background_type: str, sample_size: int
) -> np.ndarray:

    possible_index = []

    n_samples = y_data.shape[0]
    total_n_pixels = y_data.shape[1] * y_data.shape[2]

    if background_type == "balanced":
        min_cutoff, max_cutoff = (0.4 * total_n_pixels), (0.6 * total_n_pixels)
    if background_type == "karst":
        min_cutoff, max_cutoff = (0.5 * total_n_pixels), (total_n_pixels)
    if background_type == "non-karst":
        min_cutoff, max_cutoff = 0, (0.5 * total_n_pixels)

    for index in range(n_samples):
        sample = y_data[index]
        if (sample.sum() > min_cutoff) and (sample.sum() < max_cutoff):
            possible_index.append(index)

    chosen_index = np.random.choice(possible_index, size=sample_size, replace=False)

    return x_data[chosen_index]
