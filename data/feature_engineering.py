import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import PowerTransformer

from utils import clip


def calculate_Hilbert_Similarity(gpr_section, gpr_attributes, sample_size):
    hilb = gpr_attributes[-1]
    sim = gpr_attributes[0]

    attributes = [hilb, sim]

    gprRows = gpr_section.shape[0]

    for attNumber in range(2):
        missingRows = int((gprRows - attributes[attNumber].shape[0]) / 2)

        if missingRows > 0:
            attributes[attNumber] = np.pad(
                attributes[attNumber],
                ((missingRows, missingRows), (0, 0)),
                constant_values=0,
            )
        attributes[attNumber] = clip(attributes[attNumber], sample_size)

    hilb_sim = np.divide(attributes, attributes[1])

    hilb_sim = np.nan_to_num(hilb_sim, nan=0, posinf=0, neginf=0)

    gpr_attributes.append(hilb_sim)

    return gpr_attributes


def process_section(gpr_section, gpr_attributes, sample_size):

    gprRows = gpr_section.shape[0]
    gpr_section = clip(gpr_section, sample_size)
    gpr_section = np.expand_dims(gpr_section, -1)

    stackedAttributes = np.empty((gpr_section.shape[0], gpr_section.shape[1], 0))
    for attribute in gpr_attributes:

        missingRows = int((gprRows - attribute.shape[0]) / 2)
        if missingRows > 0:
            attribute = np.pad(
                attribute, ((missingRows, missingRows), (0, 0)), constant_values=0
            )
        attribute = clip(attribute, sample_size)
        attribute = np.expand_dims(attribute, -1)

        stackedAttributes = np.concatenate([stackedAttributes, attribute], axis=-1)

    section = np.concatenate([gpr_section, stackedAttributes], axis=-1)

    return section


def normalize_data(sectionsList, min_max_scale=False):

    yj_list = []
    for channel in range(sectionsList[0].shape[-1]):
        yj = PowerTransformer(method="yeo-johnson")
        array = np.concatenate(
            [section[..., channel].reshape(-1) for section in sectionsList]
        )

        # Calculate data statistics and transform data
        yj.fit(array.reshape(-1, 1))
        min = array.min()
        max = array.max()
        yj_list.append(yj)

        for idx in range(len(sectionsList)):
            original_shape = sectionsList[idx][..., channel].shape
            sectionsList[idx][..., channel] = yj.transform(
                sectionsList[idx][..., channel].reshape(-1, 1)
            ).reshape(original_shape)

    if min_max_scale:
        for channel in range(sectionsList[0].shape[-1]):
            array = np.concatenate(
                [section[..., channel].reshape(-1) for section in sectionsList]
            )
            min = array.min()
            max = array.max()
            for idx in range(len(sectionsList)):
                sectionsList[idx][..., channel] = (
                    sectionsList[idx][..., channel] - min
                ) / (max - min)

    return sectionsList, yj_list


def data_augmentation(
    array: npt.NDArray, loc: float | None, mu: float | None, add_noise: bool = False
) -> npt.NDArray:

    if add_noise:
        array_aug = np.vstack(
            [
                array,
                np.flip(array, axis=0) + np.random.normal(loc, mu, size=(array.shape)),
                np.flip(array, axis=1) + np.random.normal(loc, mu, size=(array.shape)),
                np.rot90(array, k=1, axes=(1, 2))
                + np.random.normal(loc, mu, size=(array.shape)),
                np.rot90(array, k=2, axes=(1, 2))
                + np.random.normal(loc, mu, size=(array.shape)),
                np.rot90(array, k=3, axes=(1, 2))
                + np.random.normal(loc, mu, size=(array.shape)),
            ]
        )

        array_aug = np.where(array_aug < 0, 0, array_aug)
        array_aug = np.where(array_aug > 1, 1, array_aug)

    else:

        array_aug = np.vstack(
            [
                array,
                np.flip(array, axis=0),
                np.flip(array, axis=1),
                np.rot90(array, k=1, axes=(1, 2)),
                np.rot90(array, k=2, axes=(1, 2)),
                np.rot90(array, k=3, axes=(1, 2)),
            ]
        )
    return array_aug
