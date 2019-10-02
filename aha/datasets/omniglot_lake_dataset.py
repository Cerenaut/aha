# Copyright (C) 2019 Project AGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""OmniglotLakeDataset class."""

import os
import math
import zipfile
import tempfile
import logging

from random import shuffle
from six.moves import urllib

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset
from pagi.utils.tf_utils import tf_invert_values, tf_centre_of_mass


class OmniglotLakeDataset(Dataset):
  """
  Omniglot Dataset assembled in a way that can be used for the Lake test
  i.e. unique exemplars, followed by unique exemplars of same classes in same order
  """

  # Mapping of alphabets (superclasses) to characters (classes) populated when dataset is loaded
  CLASS_MAP = {}

  def __init__(self, directory, batch_size, test_classes, instance_mode):
    super(OmniglotLakeDataset, self).__init__(
        name='omniglot',
        directory=directory,
        dataset_shape=[-1, 105, 105, 1],
        train_size=19280,
        test_size=13180,
        num_train_classes=964,
        num_test_classes=659,
        num_classes=1623)

    self._batch_size = batch_size
    self._test_classes = test_classes
    self._instance_mode = instance_mode

    self._dataset_show_files = []
    self._dataset_show_labels = []
    self._dataset_match_files = []
    self._dataset_match_labels = []

  def set_shape(self, height, width):
    self._dataset_shape[1] = height
    self._dataset_shape[2] = width

  def get_train(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot training data.
    """

    if len(self._dataset_show_files) == 0:
      self._create_test_sets()
    return self._dataset(self._dataset_show_files, self._dataset_show_labels)

  def get_test(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot test data.
    """
    if len(self._dataset_match_files) == 0:
      self._create_test_sets()
    return self._dataset(self._dataset_match_files, self._dataset_match_labels)

  def _create_test_sets(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot test data.

    The order of samples is such that they are divided into batches,
    and always have one of each of the first `batch_size` classes from `test_show_classes`
    """

    # assert that batch_size <= test_classes
    # assert that batches <= len(labels) / batch_size

    # 1) Get full list of possible samples, filename and label
    # ------------------------------------------------------------------------------------
    images_folder = self._download(self._directory, 'images_background')
    files, labels = self._filenames_and_labels(images_folder)

    # filter the filenames, labels by the list in the_classes
    the_classes = self.get_classes_by_superclass(self._test_classes)

    files_filtered = []
    labels_filtered = []
    for file, label in zip(files, labels):
      if label in the_classes:
        files_filtered.append(file)
        labels_filtered.append(label)
    files = files_filtered
    labels = labels_filtered

    # 2) Sort the full list 'labels' into batches of unique classes
    # ------------------------------------------------------------------------------------
    self._dataset_show = []
    self._dataset_match = []

    # first shuffle the order
    dataset = list(zip(files, labels))
    np.random.shuffle(dataset)
    files, labels = map(list, zip(*dataset))

    # then repeatedly sample with removal, assembling all batches
    data_show = []
    data_match = []

    end_batches = False
    batch_num = -1
    while not end_batches:
      batch_num += 1

      if batch_num >= 20:
        break

      # build a new batch
      batch_labels = []
      batches_labels = []
      batch_label_index = -1
      batch_label = ''
      for i, sample in enumerate(range(self._batch_size)):

        # if instance mode, then we only want one class, repeated batch_size times for train.
        # 'test' should be a copy (but usually done in workflow anyway)
        if self._instance_mode:

          # get the next index
          # ----------------------------

          index = -1
          # first item in batch, sample a random label that has not been chosen previously
          if batch_label_index == -1:
            # select first sample that is not in all batches so far (we want each batch to be a unique class)
            for idx, label in enumerate(labels):
              if label not in batches_labels:
                batch_label_index = idx
                batch_label = label
                batches_labels.append(batch_label)  # remember which labels we added to all batches
                index = idx
                break

            logging.debug("====================== Batch={}, label={}".format(batch_num, batch_label))

          # from then on, choose another exemplar from the same class
          else:
            # select same class for a 'match' sample
            if batch_label in labels:
              index = labels.index(batch_label)

          logging.debug("==================     ----> Batch={}, index={}".format(batch_num, index))

          # detect reaching the end of the dataset i.e. not able to assemble a new batch
          if index == -1:
            logging.info('Not able to find a unique class to assemble a new batch, '
                         'on batch={0}, sample={1}'.format(batch_num, sample))
            end_batches = True
            break

          # add to the datasets
          file = files.pop(index)
          label = labels.pop(index)
          data_show.append([file, label])
          data_match.append([file, label])

        else:
          # select first sample that is not in batch so far (to get unique)
          index = -1
          for idx, label in enumerate(labels):
            if label not in batch_labels:
              index = idx
              break

          # detect reaching the end of the dataset i.e. not able to assemble a new batch
          if index == -1:
            logging.info('Not able to find a unique class to assemble a new batch, '
                         'on batch={0}, sample={1}'.format(batch_num, sample))
            end_batches = True
            break

          # add to the 'show' dataset
          file = files.pop(index)
          label = labels.pop(index)
          data_show.append([file, label])

          batch_labels.append(label)   # remember which labels we added to this batch

          # select same class for a 'match' sample
          index = labels.index(label)

          # add to the 'match' dataset
          label = labels.pop(index)
          file = files.pop(index)
          data_match.append([file, label])

    # convert from array of pairs, to pair of arrays
    self._dataset_show_files, self._dataset_show_labels = map(list, zip(*data_show))
    self._dataset_match_files, self._dataset_match_labels = map(list, zip(*data_match))

  def get_classes_by_superclass(self, superclasses, proportion=1.0):
    """
    Retrieves a proportion of classes belonging to a particular superclass, defaults to retrieving all classes
    i.e. proportion=1.0.

    Arguments:
      superclasses: A single or list of the names of superclasses, or a single name of a superclass.
      proportion: A float that indicates the proportion of sub-classes to retrieve (default=1.0)
    """
    if not self.CLASS_MAP:
      raise ValueError('Superclass to class mapping (CLASS_MAP) is not populated yet.')

    def filter_classes(classes, proportion, do_shuffle=True):
      """Filters the list of classes by retrieving a proportion of shuffled classes."""
      if do_shuffle:
        shuffle(classes)
      num_classes = math.ceil(len(classes) * float(proportion))
      return classes[:num_classes]

    classes = []
    if superclasses is None or (isinstance(superclasses, list) and len(superclasses) == 0):
      for superclass in self.CLASS_MAP.keys():
        subclasses = filter_classes(self.CLASS_MAP[superclass], proportion)
        classes.extend(subclasses)
    elif isinstance(superclasses, list):
      for superclass in superclasses:
        subclasses = filter_classes(self.CLASS_MAP[superclass], proportion)
        classes.extend(subclasses)
    else:   # string - single superclass specified
      classes = filter_classes(self.CLASS_MAP[superclasses], proportion)

    return classes

  def _dataset(self, filenames, labels):

    def parse_function(filenames, label):
      return OmniglotLakeDataset.parse_function(filenames, label, self.shape, self._dataset_shape, centre=True)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    return dataset

  def _dataset_by_filename(self, directory, images_file):
    """Download and parse Omniglot dataset."""
    images_folder = self._download(directory, images_file)
    filenames, labels = self._filenames_and_labels(images_folder)
    dataset = self._dataset(filenames, labels)
    return dataset

  def _download(self, directory, filename):
    """Download (and unzip) a file from the Omniglot dataset if not already done."""
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)
    if tf.gfile.Exists(filepath):
      return filepath
    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    url = 'https://github.com/brendenlake/omniglot/raw/master/python/' + (
        filename + '.zip')
    _, zipped_filepath = tempfile.mkstemp(suffix='.zip')
    logging.info('Downloading %s to %s', url, zipped_filepath)
    urllib.request.urlretrieve(url, zipped_filepath)

    zip_ref = zipfile.ZipFile(zipped_filepath, 'r')
    zip_ref.extractall(dirpath)
    zip_ref.close()

    os.remove(zipped_filepath)
    return filepath

  def _filenames_and_labels(self, image_folder):
    """Get the image filename and label for each Omniglot character."""
    # Compute list of characters (each is a folder full of images)
    character_folders = []
    for family in os.listdir(image_folder):
      if os.path.isdir(os.path.join(image_folder, family)):
        append_characters = False
        if family not in self.CLASS_MAP:
          self.CLASS_MAP[family] = []
          append_characters = True
        for character in os.listdir(os.path.join(image_folder, family)):
          character_folder = os.path.join(image_folder, family, character)
          if append_characters and os.path.isdir(character_folder):
            character_file = os.listdir(character_folder)[0]
            character_label = int(character_file.split('_')[0])
            self.CLASS_MAP[family].append(character_label)
          character_folders.append(character_folder)
      else:
        logging.warning('Path to alphabet is not a directory: %s', os.path.join(image_folder, family))

    # Count number of images
    num_images = 0
    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          num_images += 1

    # Put them in one big array, and one for labels
    #   A 4D uint8 numpy array [index, y, x, depth].
    idx = 0
    filename_arr = []
    label_arr = np.zeros([num_images], dtype=np.int32)

    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          filename_arr.append(os.path.join(path, file))
          label_arr[idx] = file.split('_')[0]
          idx += 1

    return filename_arr, label_arr

  @staticmethod
  def parse_function(filename, label, shape, dataset_shape, centre=True):

    if centre:
      """Read and parse the image from a filepath."""
      image_string = tf.read_file(filename)

      # Don't use tf.image.decode_image, or the output shape will be undefined
      image = tf.image.decode_jpeg(image_string, channels=shape[3])

      # This will convert to float values in [0, 1] result shape ?,?,1
      image = tf.image.convert_image_dtype(image, tf.float32)

      # Resize image
      image = tf.image.resize_images(image, [shape[1], shape[2]])

      # Invert foreground/background so digit is 1 and background is 0
      image = tf_invert_values(image, None)

      # Centre image
      centre = [shape[1] * 0.5, shape[2] * 0.5]
      centre_of_mass = tf_centre_of_mass([image], [1, shape[1], shape[2], 1])
      translation = centre - centre_of_mass  # e.g. Com = 27, centre = 25, 25-27 = -2

      # Translate [dx, dy]
      image = tf.contrib.image.translate([image], [translation], interpolation='BILINEAR')

      # flatten feature dimension
      image = tf.reshape(image, dataset_shape[1:])

      return image, label

    else:
      """Read and parse the image from a filepath."""
      image_string = tf.read_file(filename)

      # Don't use tf.image.decode_image, or the output shape will be undefined
      image = tf.image.decode_jpeg(image_string, channels=shape[3])

      # This will convert to float values in [0, 1]
      image = tf.image.convert_image_dtype(image, tf.float32)

      # Resize image and flatten feature dimension
      image = tf.image.resize_images(image, [shape[1], shape[2]])
      image = tf.reshape(image, dataset_shape[1:])

      return image, label
