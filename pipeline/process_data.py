import datetime
import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from gpr_unet import load_data, utils


@hydra.main(config_path=".", config_name="config.yaml")
def main(config: DictConfig):

    raw_data_path = script_dir / "raw_data"

    sample_size = config["sampling"]["sample_size"]

    for number in range(89, 97):
        logger.info("Processing section %d", number)

        logger.info("Loading data and converting to numpy arrays")
        gpr = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/gpr_sections/FILE0{number}.SGY"
        )
        logger.debug("GPR shape: %s", gpr.shape)
        similarity = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/attributes/Sim{number}.dat"
        )
        logger.debug("Similarity data shape: %s", similarity.shape)
        energy = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/attributes/En{number}.dat"
        )
        logger.debug("Energy data shape: %s", energy.shape)
        inst_freq = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/attributes/InstFreq{number}.dat"
        )
        logger.debug("Inst freq data shape: %s", inst_freq.shape)
        inst_phase = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/attributes/InstPha{number}.dat"
        )
        logger.debug("Inst phase data shape: %s", inst_phase.shape)
        hilbert = load_data.gpr_data_to_numpy(
            f"{raw_data_path}/attributes/Hilb{number}.dat"
        )
        logger.debug("Hilbert data shape: %s", hilbert.shape)

        logger.info(
            "Clipping the GPR section to a format divisible by the sample size: %s",
            sample_size,
        )

        initial_number_of_rows = gpr.shape[0]
        gpr = utils.clip(gpr, sample_size)
        logger.info("GPR shape: %s", gpr.shape)

        logger.info("Padding and clipping data attributes to the same size")
        similarity = load_data.pad_attribute(similarity, initial_number_of_rows)
        similarity = utils.clip(similarity, sample_size)
        logger.debug("Similarity data shape: %s", similarity.shape)
        energy = load_data.pad_attribute(energy, initial_number_of_rows)
        energy = utils.clip(energy, sample_size)
        logger.debug("Energy data shape: %s", energy.shape)
        inst_freq = load_data.pad_attribute(inst_freq, initial_number_of_rows)
        inst_freq = utils.clip(inst_freq, sample_size)
        logger.debug("Inst freq data shape: %s", inst_freq.shape)
        inst_phase = load_data.pad_attribute(inst_phase, initial_number_of_rows)
        inst_phase = utils.clip(inst_phase, sample_size)
        logger.debug("Inst phase data shape: %s", inst_phase.shape)
        hilbert = load_data.pad_attribute(hilbert, initial_number_of_rows)
        hilbert = utils.clip(hilbert, sample_size)
        logger.debug("Hilbert data shape: %s", hilbert.shape)

        logger.info("Calculating the Hilbert Trace/Similarity")
        hilbert_similarity = np.divide(hilbert, similarity)
        hilbert_similarity = np.nan_to_num(
            hilbert_similarity, nan=0, posinf=0, neginf=0
        )

        logger.info("Loading the ground truth and clipping it")
        ground_truth = load_data.load_ground_truth(
            f"{raw_data_path}/ground_truth/0{number}.jpg"
        )
        ground_truth = utils.clip(ground_truth, sample_size)

        logger.info("Concatenating data into a single array")
        data = np.stack(
            [gpr, similarity, energy, inst_freq, inst_phase, hilbert_similarity],
            axis=-1,
        )

        logger.info("Saving data to disk")
        logger.info("X shape: %s", data.shape)
        logger.info("Y shape: %s", ground_truth.shape)
        np.save(output_path.joinpath(f"x_data_{number}.npy"), data)
        np.save(output_path.joinpath(f"y_data_{number}.npy"), ground_truth)


if __name__ == "__main__":

    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    experiment_name = f"{year}-{month}-{day}-{hour}-{minute}"
    mlflow.start_run(run_name=experiment_name, nested=True)

    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir.joinpath("processed_data/")

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename="process_data.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("outputs/process_data.log"))
