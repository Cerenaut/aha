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

"""OmniglotUnseenOneShotDataset class."""

import os
import math
import zipfile
import tempfile
import logging

from random import shuffle
from six.moves import urllib

import numpy as np
import tensorflow as tf

from pagi.utils.tf_utils import tf_invert_values, tf_centre_of_mass

from aha.datasets.omniglot_lake_dataset import OmniglotLakeDataset
from aha.datasets.omniglot_unseen_dataset import OmniglotUnseenDataset


class OmniglotUnseenOneShotDataset(OmniglotUnseenDataset):
  """
  Omniglot Dataset assembled in a way that can be used for the Lake test
  i.e. unique exemplars, followed by unique exemplars of same classes in same order
  """

  # Mapping of alphabets (superclasses) to characters (classes) populated when dataset is loaded
  CLASS_MAP = {}

  def __init__(self, directory, batch_size):
    super(OmniglotUnseenOneShotDataset, self).__init__(directory=directory)

    self._batch_size = batch_size

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

    train_images_folder = self._download(self._directory, 'images_evaluation_supervised/train')
    test_images_folder = self._download(self._directory, 'images_evaluation_supervised/test')
    unseen_images_folder = self._download(self._directory, 'images_evaluation_unseen')

    train_files, train_labels = self._filenames_and_labels(train_images_folder)
    test_files, test_labels = self._filenames_and_labels(test_images_folder)
    unseen_files, unseen_labels = self._filenames_and_labels(unseen_images_folder)

    # 2) Sort the full list 'labels' into batches of unique classes
    # ------------------------------------------------------------------------------------
    self._dataset_show = []
    self._dataset_match = []

    # first shuffle the order
    def shuffle(files, labels):
      dataset = list(zip(files, labels))
      np.random.shuffle(dataset)
      files, labels = map(list, zip(*dataset))
      return files, labels

    train_files, train_labels = shuffle(train_files, train_labels)
    test_files, test_labels = shuffle(test_files, test_labels)
    unseen_files, unseen_labels = shuffle(unseen_files, unseen_labels)

    print(np.unique(unseen_labels), '\n')
    print(np.unique(train_labels), '\n')
    print(np.unique(test_labels), '\n')

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
        show_files, show_labels = train_files, train_labels
        match_files, match_labels = test_files, test_labels
        # files, labels = train_files, train_files

        # first element is unseen
        if i == 0:
          show_files, show_labels = unseen_files, unseen_labels
          match_files, match_labels = unseen_files, unseen_labels
          # files, labels = unseen_files, unseen_labels

        # select first sample that is not in batch so far (to get unique)
        index = -1
        for idx, label in enumerate(show_labels):
          if label not in batch_labels:
            index = idx
            break

        # detect reaching the end of the dataset i.e. not able to assemble a new batch
        if index == -1:
          logging.info('Not able to find a unique class to assemble a new batch, '
                       'on batch={0}, sample={1}'.format(batch_num, sample))
          end_batches = True
          break

        try:
          # add to the 'show' dataset
          file = show_files.pop(index)
          label = show_labels.pop(index)
          data_show.append([file, label])

          batch_labels.append(label)   # remember which labels we added to this batch

          # select same class for a 'match' sample
          index = match_labels.index(label)

          # add to the 'match' dataset
          file = match_files.pop(index)
          label = match_labels.pop(index)
          data_match.append([file, label])
        except ValueError:
          logging.info('Not able to find a unique sample for this class to assemble this batch, '
                       'on batch={0}, sample={1}'.format(batch_num, sample))
          # end_batches = True
          break

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