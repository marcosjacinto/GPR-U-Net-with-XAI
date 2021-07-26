import segyio
import tensorflow as tf
import numpy as np
from skimage import color, io
from sklearn.model_selection import train_test_split
import glob
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer


def load_section(section_number, x_path, y_path, attribute_path):

  x = segy_to_numpy(load_segy(x_path + 'FILE0' + str(section_number) + '.SGY'))
  y = load_ground_truth(y_path + '0' + str(section_number) + '.jpg')
  attributes = []
  for att in ['Sim', 'En', 'InstFreq', 'InstPha', 'Hilb']:
    attributes.append(np.genfromtxt(attribute_path + att + str(section_number) + '.dat').T)

  return x, y, attributes

def load_ground_truth(image_path):

  array = color.rgb2gray(io.imread(image_path))
  array = np.where(array >= 0.5, 1, 0)

  return array

def load_segy(path):

  segy = segyio.open(path, ignore_geometry = True)

  return segy

def segy_to_numpy(segy):

  array = np.empty((len(segy.samples), len(segy.trace)))

  for n in range(0, len(segy.trace)):
    array[:, n] = segy.trace[n]

  return array

def clip(array, sample_size):
  array = array[:((array.shape[-2] // sample_size) * sample_size), : ((array.shape[-1] // sample_size) * sample_size)]

  return array

def sample_input_data(array, sample_size, step_div = 1):

  total = 0
  step = int(sample_size/step_div)
  shape1, shape2, shape3 = array.shape[0], array.shape[1], array.shape[2]

  for i in range(int(shape1/(step)) - (step_div - 1)):
    for w in range(int(shape2/(step)) - (step_div - 1)):
      total += 1

  x_sampled = np.empty((total, sample_size, sample_size, shape3))
  

  n = 0
  for i in range(int(shape1/(step)) - (step_div - 1)):
    istart = int(step * i)
    iend = int(istart + sample_size)
    for w in range(int(shape2/(step)) - (step_div - 1)):
      wstart = int(step * w)
      wend = int(wstart + sample_size)
      x_sampled[n, ...] = array[istart : iend, wstart : wend].reshape(1, sample_size, sample_size, shape3)
      n += 1

  return x_sampled

def sample_ground_truth(array, sample_size, step_div = 1):

  array = clip(array, 16)

  total = 0
  step = int(sample_size/step_div)
  shape1, shape2 = array.shape[0], array.shape[1]

  for i in range(int(shape1/(step)) - (step_div - 1)):
    for w in range(int(shape2/(step)) - (step_div - 1)):
      total += 1

  y_sampled = np.empty((total, sample_size, sample_size))

  n = 0
  for i in range(int(shape1/(step)) - (step_div - 1)):
    istart = int(step * i)
    iend = int(istart + sample_size)
    for w in range(int(shape2/(step)) - (step_div - 1)):
      wstart = int(step * w)
      wend = int(wstart + sample_size)
      y_sampled[n, ...] = array[istart : iend, wstart : wend].reshape(1, sample_size, sample_size)
      n += 1

  return y_sampled

def calculate_Hilbert_Similarity(gprSection, gprAttributes, sample_size):
  hilb = gprAttributes[-1]
  sim = gprAttributes[0]

  attributes = [hilb, sim]

  gprRows = gprSection.shape[0]

  for attNumber in range(2):
    missingRows = int((gprRows - attributes[attNumber].shape[0]) / 2)
 
    if missingRows > 0:
      attributes[attNumber] = np.pad(attributes[attNumber],
                                    ((missingRows, missingRows), (0, 0)),
                                    constant_values = 0)
    attributes[attNumber] = clip(attributes[attNumber], sample_size)

  hilb_sim = attributes[0] / attributes[1]

  hilb_sim = np.nan_to_num(hilb_sim, nan = 0, posinf = 0, neginf = 0)

  gprAttributes.append(hilb_sim)

  return gprAttributes

def process_section(gprSection, gprAttributes, sample_size):
  
  gprRows = gprSection.shape[0]
  gprSection = clip(gprSection, sample_size)
  gprSection = np.expand_dims(gprSection, -1)
  
  stackedAttributes = np.empty((gprSection.shape[0], gprSection.shape[1], 0))
  for attribute in gprAttributes:

    missingRows = int((gprRows - attribute.shape[0]) / 2)
    if missingRows > 0:
      attribute = np.pad(attribute, ((missingRows, missingRows), (0, 0)), constant_values = 0)
    attribute = clip(attribute, sample_size)
    attribute = np.expand_dims(attribute, -1)

    stackedAttributes = np.concatenate([stackedAttributes, attribute], axis = -1)

  section = np.concatenate([gprSection, stackedAttributes], axis = -1)

  return section

def normalize_data(sectionsList, min_max_scale = False):

  yj_list = []
  for channel in range(sectionsList[0].shape[-1]):
    yj = PowerTransformer(method='yeo-johnson')
    array = np.concatenate([section[..., channel].reshape(-1) for section in sectionsList])

    # Calculate data statistics and transform data
    yj.fit(array.reshape(-1, 1))
    min = array.min()
    max = array.max()
    yj_list.append(yj)

    for idx in range(len(sectionsList)):
      original_shape = sectionsList[idx][..., channel].shape
      sectionsList[idx][..., channel] = yj.transform(sectionsList[idx][..., channel].reshape(-1, 1)).reshape(original_shape)

  if min_max_scale:
    for channel in range(sectionsList[0].shape[-1]):
      array = np.concatenate([section[..., channel].reshape(-1) for section in sectionsList])
      min = array.min()
      max = array.max()
      for idx in range(len(sectionsList)):
        sectionsList[idx][..., channel] = (sectionsList[idx][..., channel] - min) / (max - min)

  return sectionsList, yj_list

def create_train_dataset(xList, yList, valSize = 0.15,
                         augment = False, add_noise = False, loc = 0, mu = 0.01):

  x_train = np.concatenate(xList, axis = 0)
  y_train = np.concatenate(yList, axis = 0)

  x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size = valSize)
  if augment:
    x_train = data_augmentation(x_train, loc = loc, mu = mu, add_noise = add_noise)
    y_train = data_augmentation(y_train, loc = None, mu = None)

  return x_train, x_val, y_train, y_val

def data_augmentation(array, loc, mu, add_noise = False):

  if add_noise:
    array_aug = np.vstack([array,
                         np.flip(array, axis = 0) + np.random.normal(loc, mu, size = (array.shape)),
                         np.flip(array, axis = 1) + np.random.normal(loc, mu, size = (array.shape)),
                         np.rot90(array, k = 1, axes = (1, 2)) + np.random.normal(loc, mu, size = (array.shape)),
                         np.rot90(array, k = 2, axes = (1, 2)) + np.random.normal(loc, mu, size = (array.shape)),
                         np.rot90(array, k = 3, axes = (1, 2)) + np.random.normal(loc, mu, size = (array.shape))]
                        )
    
    array_aug = np.where(array_aug < 0, 0, array_aug)
    array_aug = np.where(array_aug > 1, 1, array_aug)

  else:
    
    array_aug = np.vstack([array,
                         np.flip(array, axis = 0),
                         np.flip(array, axis = 1),
                         np.rot90(array, k = 1, axes = (1, 2)),
                         np.rot90(array, k = 2, axes = (1, 2)),
                         np.rot90(array, k = 3, axes = (1, 2))]
                        )
  return array_aug