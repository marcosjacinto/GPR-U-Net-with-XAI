from pickle import dump


from auxiliary_functions import *


input_data_path = 'GPR-U-Net-with-XAI/data/original/gpr_sections/'
ground_truth_path = 'GPR-U-Net-with-XAI/data/original/ground_truth/'
attributes_path = 'GPR-U-Net-with-XAI/data/original/attributes/'
output_path = 'GPR-U-Net-with-XAI/data/processed/'

x89, y89, x89attributes = load_section(89, input_data_path, ground_truth_path, attributes_path)
x90, y90, x90attributes = load_section(90, input_data_path, ground_truth_path, attributes_path)
x91, y91, x91attributes = load_section(91, input_data_path, ground_truth_path, attributes_path)
x92, y92, x92attributes = load_section(92, input_data_path, ground_truth_path, attributes_path)
x93, y93, x93attributes = load_section(93, input_data_path, ground_truth_path, attributes_path)
x94, y94, x94attributes = load_section(94, input_data_path, ground_truth_path, attributes_path)
x95, y95, x95attributes = load_section(95, input_data_path, ground_truth_path, attributes_path)
x96, y96, x96attributes = load_section(96, input_data_path, ground_truth_path, attributes_path)

sample_size = 16

x89attributes = calculate_Hilbert_Similarity(x89, x89attributes, sample_size)
x90attributes = calculate_Hilbert_Similarity(x90, x90attributes, sample_size)
x91attributes = calculate_Hilbert_Similarity(x91, x91attributes, sample_size)
x92attributes = calculate_Hilbert_Similarity(x92, x92attributes, sample_size)
x93attributes = calculate_Hilbert_Similarity(x93, x93attributes, sample_size)
x94attributes = calculate_Hilbert_Similarity(x94, x94attributes, sample_size)
x95attributes = calculate_Hilbert_Similarity(x94, x95attributes, sample_size)
x96attributes = calculate_Hilbert_Similarity(x95, x96attributes, sample_size)

x89full = process_section(x89, x89attributes, sample_size)
x90full = process_section(x90, x90attributes, sample_size)
x91full = process_section(x91, x91attributes, sample_size)
x92full = process_section(x92, x92attributes, sample_size)
x93full = process_section(x93, x93attributes, sample_size)
x94full = process_section(x94, x94attributes, sample_size)
x95full = process_section(x95, x95attributes, sample_size)
x96full = process_section(x96, x96attributes, sample_size)

sections = [x89full, x90full, x91full, x92full, x93full, x94full, x95full, x96full]
min_max_scale = True
((x89norm, x90norm, x91norm, x92norm, x93norm, x94norm, x95norm, x96norm), yj_list) = normalize_data(sections, min_max_scale = min_max_scale)

dump(yj_list, open(output_path + 'yj_transformer_list.pkl', 'wb')) # saves the power transformer

sample_density = 1 # the higher the more samples will be created using sliding windows

x89sampled = sample_input_data(x89norm, sample_size, sample_density)
x90sampled = sample_input_data(x90norm, sample_size, sample_density)
x91sampled = sample_input_data(x91norm, sample_size, sample_density)
x92sampled = sample_input_data(x92norm, sample_size, sample_density)
x93sampled = sample_input_data(x93norm, sample_size, sample_density)
x94sampled = sample_input_data(x94norm, sample_size, sample_density)
x95sampled = sample_input_data(x95norm, sample_size, sample_density)
x96sampled = sample_input_data(x96norm, sample_size, sample_density)

y89sampled = sample_ground_truth(y89, sample_size, sample_density)
y90sampled = sample_ground_truth(y90, sample_size, sample_density)
y91sampled = sample_ground_truth(y91, sample_size, sample_density)
y92sampled = sample_ground_truth(y92, sample_size, sample_density)
y93sampled = sample_ground_truth(y93, sample_size, sample_density)
y94sampled = sample_ground_truth(y94, sample_size, sample_density)
y95sampled = sample_ground_truth(y95, sample_size, sample_density)
y96sampled = sample_ground_truth(y96, sample_size, sample_density)


xList = [x90sampled, x91sampled, x92sampled, x93sampled, x94sampled, x95sampled]
yList = [y90sampled, y91sampled, y92sampled, y93sampled, y94sampled, y95sampled]

loc, sigma = 0, 0.01
noise = True
augment = True


x_train, x_val, y_train, y_val = create_train_dataset(xList, yList,
                                                      augment = augment,
                                                      add_noise = noise,
                                                      loc = loc,
                                                      mu = sigma)

x_test, y_test = np.concatenate([x89sampled, x96sampled], axis = 0), np.concatenate([y89sampled, y96sampled], axis = 0)

x_train.dtype = 'float16'
y_train.dtype  = 'float16'
x_val.dtype = 'float16'
y_val.dtype  = 'float16'
x_test.dtype  = 'float16'
y_test.dtype  = 'float16'
x89norm.dtype = 'float16'
x96norm.dtype = 'float16'
x89.dtype = 'float16'
x96.dtype = 'float16'

with open(output_path + 'dataset_description.txt', 'w') as f:
  f.write('Dataset description:\n')
  f.write('This dataset contains the following attributes in order inside the array:\n')
  # Modificar aqui caso sejam adicionados mais atributos
  f.write('Similarity, Energy, Instantaneous Frequency,\n') 
  f.write('Instantaneous Phase, Hilbert/Similarity\n')
  f.write(f'Number of examples in the train dataset {x_train.shape[0]}\n')
  f.write(f'Number of examples in the validation dataset {x_val.shape[0]}\n')
  f.write(f'Number of examples in the test dataset {x_test.shape[0]}\n')
  # Modificar aqui caso altere a seção de teste
  f.write('Test section was section number 089 and 096\n')
  f.write(f'Sample size is {sample_size}x{sample_size}\n')
  f.write(f'Sample density is {sample_density}\n')
  if noise:
    f.write('Noise was used\n')
    f.write(f'Noise parameters used were: {loc} and {sigma}\n')
  else:
    f.write('No noise was added\n')
  if min_max_scale:
    f.write('Scaled between 0 and 1\n')

np.save(output_path + 'x_train.npy', x_train)
np.save(output_path + 'y_train.npy', y_train)
np.save(output_path + 'x_val.npy', x_val)
np.save(output_path + 'y_val.npy', y_val)
np.save(output_path + 'x_test.npy', x_test)
np.save(output_path + 'y_test.npy', y_test)
np.save(output_path + 'x_test_section_89.npy', x89norm)
np.save(output_path + 'y_test_section_89.npy', y89)
np.save(output_path + 'x_test_section_96.npy', x96norm)
np.save(output_path + 'y_test_section_96.npy', y96)