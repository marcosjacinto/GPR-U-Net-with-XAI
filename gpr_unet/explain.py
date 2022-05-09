from pathlib import Path

import mlflow
import numpy as np
import shap
import tensorflow as tf

# shap.explainers._deep.deep_tf.op_handlers[
#     "Conv2DBackpropInput"
# ] = shap.explainers._deep.deep_tf.passthrough
# tf.compat.v1.disable_v2_behavior()


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


def get_kernel_explainer(
    run_id: str, background_samples: np.ndarray
) -> shap.KernelExplainer:

    run = mlflow.get_run(run_id)
    artifact_uri = Path(run.info.artifact_uri)
    model_path = artifact_uri / "model/data/model"

    model = tf.keras.models.load_model(model_path)

    return shap.KernelExplainer(model, background_samples)
