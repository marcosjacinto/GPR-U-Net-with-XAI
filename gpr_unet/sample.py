import numpy as np


def sample_data(
    array: np.ndarray, sample_size: int, sampling_step: int = 1
) -> np.ndarray:

    step = int(sample_size / sampling_step)
    rows, columns, channels = array.shape[0], array.shape[1], array.shape[2]
    
    max_rows = int(rows / step) - (sampling_step - 1)
    max_columns = int(columns / step) - (sampling_step - 1)

    if sampling_step == 1:
        total_samples = int(rows * columns / (sample_size * sample_size))
    else:
        total_samples = 0
        for i in range(max_rows):
            for w in range(max_columns):
                total_samples += 1
                
    sampled_data = np.empty((total_samples, sample_size, sample_size, channels))

    current_sample = 0

    for row_index in range(max_rows):

        initial_row = int(step * row_index)
        final_row = int(initial_row + sample_size)

        for column_index in range(max_columns):

            initial_column = int(step * column_index)
            final_column = int(initial_column + sample_size)

            sample = array[initial_row:final_row, initial_column:final_column]
            sample = sample.reshape(1, sample_size, sample_size, channels)
            sampled_data[current_sample, ...] = sample
            current_sample += 1

    return sampled_data

def reconstruct_predictions(
    predictions: np.ndarray, original_shape: tuple, sample_size: int, sampling_step: int = 1, 
) -> np.ndarray:
    
    step = int(sample_size / sampling_step)
    rows, columns, channels = original_shape[0], original_shape[1], original_shape[2]
    reconstructed_predictions = np.empty((rows, columns, 1))

    max_rows = int(rows / step) - (sampling_step - 1)
    max_columns = int(columns / step) - (sampling_step - 1)

    if sampling_step == 1:
        total_samples = int(rows * columns / (sample_size * sample_size))
    else:
        total_samples = 0
        for i in range(max_rows):
            for w in range(max_columns):
                total_samples += 1

    current_sample = 0
    
    for row_index in range(max_rows):

        initial_row = int(step * row_index)
        final_row = int(initial_row + sample_size)

        for column_index in range(max_columns):

            initial_column = int(step * column_index)
            final_column = int(initial_column + sample_size)
            
            reconstructed_predictions[
                initial_row:final_row, initial_column:final_column
            ] = predictions[current_sample]
            
            current_sample += 1

    return reconstructed_predictions