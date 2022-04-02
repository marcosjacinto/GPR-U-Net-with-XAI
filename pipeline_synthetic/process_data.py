import logging
from pathlib import Path

import mlflow
import numpy as np

from gpr_unet import load_data, utils


def main():

    raw_data_path = "pipeline_synthetic/raw_data"

    logger.info("Loading data and converting to numpy arrays")
    gpr = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr.sgy")
    similarity = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr_Similarity.sgy")
    energy = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr_Energy.sgy")
    inst_freq = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr_IntFreq.sgy")
    inst_phase = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr_IntPhase.sgy")
    hilbert = load_data.gpr_data_to_numpy(f"{raw_data_path}/gpr_HilbertTrace.sgy")
    logger.info("Data loaded")

    sample_size = 16  # FIXME: This should be a parameter in config

    logger.info(
        "Clipping the GPR section to a format divisible by the sample size: %s",
        sample_size,
    )
    gpr = utils.clip(gpr, sample_size)
    number_of_rows = gpr.shape[0]

    logger.info("Padding and clipping data attributes to the same size")
    similarity = load_data.pad_attribute(similarity, number_of_rows)
    similarity = utils.clip(similarity, sample_size)
    energy = load_data.pad_attribute(energy, number_of_rows)
    energy = utils.clip(energy, sample_size)
    inst_freq = load_data.pad_attribute(inst_freq, number_of_rows)
    inst_freq = utils.clip(inst_freq, sample_size)
    inst_phase = load_data.pad_attribute(inst_phase, number_of_rows)
    inst_phase = utils.clip(inst_phase, sample_size)
    hilbert = load_data.pad_attribute(hilbert, number_of_rows)
    hilbert = utils.clip(hilbert, sample_size)

    logger.info("Calculating the Hilbert Trace/Similarity")
    hilbert_similarity = np.divide(hilbert, similarity)
    hilbert_similarity = np.nan_to_num(hilbert_similarity, nan=0, posinf=0, neginf=0)

    logger.info("Concatenating data into a single array")
    data = np.stack(
        [gpr, similarity, energy, inst_freq, inst_phase, hilbert_similarity], axis=-1
    )

    logger.info("Saving data to disk")
    np.save(output_path.joinpath("data.npy"), data)


if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir.joinpath("processed_data/")

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=script_dir.joinpath("process_data.log"),
        filemode="w",
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("process_data.log"))