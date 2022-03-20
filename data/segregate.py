import numpy as np
from sklearn.model_selection import train_test_split

from feature_engineering import data_augmentation


def create_train_dataset(
    x_data: list,
    y_data: list,
    validation_size: float = 0.15,
    augment: bool = False,
    add_noise: bool = False,
    loc: float = 0.0,
    mu: float = 0.01,
):

    x_train = np.concatenate(x_data, axis=0)
    y_train = np.concatenate(y_data, axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_size
    )
    if augment:
        x_train = data_augmentation(x_train, loc=loc, mu=mu, add_noise=add_noise)
        y_train = data_augmentation(y_train, loc=None, mu=None)

    x_train = x_train.astype(np.float16)
    y_train = y_train.astype(np.float16)
    x_val = x_val.astype(np.float16)
    y_val = y_val.astype(np.float16)

    return x_train, x_val, y_train, y_val


def custom_train_test_split(
    loc,
    sigma,
    noise,
    augment,
    x_train_sampled,
    y_train_sampled,
    x_test_sampled,
    y_test_sampled,
):
    x_train, x_val, y_train, y_val = create_train_dataset(
        x_train_sampled,
        y_train_sampled,
        augment=augment,
        add_noise=noise,
        loc=loc,
        mu=sigma,
    )

    x_test = np.concatenate(x_test_sampled, axis=0).astype(np.float16)
    y_test = np.concatenate(y_test_sampled, axis=0).astype(np.float16)
    return x_train, x_val, y_train, y_val, x_test, y_test
