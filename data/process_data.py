from pickle import dump

from feature_engineering import (
    calculate_Hilbert_Similarity,
    normalize_data,
    process_section,
)
from load_data import load_section
from sample import sample_ground_truth, sample_input_data
from segregate import custom_train_test_split
from utils import save_processed_data, write_metadata


def main() -> None:

    input_data_path = "data/original/gpr_sections/"
    ground_truth_path = "data/original/ground_truth/"
    attributes_path = "data/original/attributes/"
    output_path = "data/processed/"

    loaded_sections = dict()
    sections = list()
    # Settings FIXME: turn into argparse
    sample_size = 16
    sample_density = (
        1  # the higher the more samples will be created using sliding windows
    )
    min_max_scale = True
    loc, sigma = 0.0, 0.01
    noise = True
    augment = True

    for number in range(89, 97):
        x, y, attributes = load_section(
            number, input_data_path, ground_truth_path, attributes_path
        )
        attributes = calculate_Hilbert_Similarity(x, attributes, sample_size)
        x_full = process_section(x, attributes, sample_size)

        sections.append(x_full)

        loaded_sections[number] = {"x": x, "y": y, "attributes": attributes}

    data_normalized, yj_list = normalize_data(sections, min_max_scale=min_max_scale)

    # saves the power transformers
    dump(yj_list, open(output_path + "yj_transformer_list.pkl", "wb"))

    x_train_sampled, y_train_sampled = list(), list()
    x_test_sampled, y_test_sampled = list(), list()

    for number, normalized_section in zip(range(89, 97), data_normalized):
        x, y, attributes = (
            loaded_sections[number]["x"],
            loaded_sections[number]["y"],
            loaded_sections[number]["attributes"],
        )
        sampled_x_data = sample_input_data(
            normalized_section, sample_size, sample_density
        )
        sampled_y_data = sample_ground_truth(y, sample_size, sample_density)
        if number == 89 or number == 96:
            x_test_sampled.append(sampled_x_data)
            y_test_sampled.append(sampled_y_data)
        else:
            x_train_sampled.append(sampled_x_data)
            y_train_sampled.append(sampled_y_data)

    x_train, x_val, y_train, y_val, x_test, y_test = custom_train_test_split(
        loc,
        sigma,
        noise,
        augment,
        x_train_sampled,
        y_train_sampled,
        x_test_sampled,
        y_test_sampled,
    )

    write_metadata(
        output_path,
        sample_size,
        sample_density,
        min_max_scale,
        loc,
        sigma,
        noise,
        x_train,
        x_val,
        x_test,
    )

    save_processed_data(
        output_path,
        loaded_sections,
        data_normalized,
        x_train,
        x_val,
        y_train,
        y_val,
        x_test,
        y_test,
    )


if __name__ == "__main__":
    main()
