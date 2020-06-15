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

"""OmniglotLakelikeRunsDataset class.."""

import os
import zipfile
import tempfile
import logging
import copy

from six.moves import urllib

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset
from pagi.utils.tf_utils import tf_invert_values, tf_centre_of_mass

from pagi.datasets.omniglot_dataset import OmniglotDataset
from aha.datasets.omniglot_lake_dataset import OmniglotLakeDataset


class OmniglotLakelikeRunsDataset(OmniglotDataset):
  """Omniglot Dataset based on tf.data."""

  def __init__(self, directory, batch_size, evaluate_mode):
    super(OmniglotLakelikeRunsDataset, self).__init__(directory=directory)

    self._num_train_classes = 659
    self._num_test_classes = 659
    self._num_classes = 659

    self._batch_size = batch_size
    self._evaluate_mode = evaluate_mode
    self._exp_mode = None
    self._run_num = None
    self.eval_classes = []

    compatible_modes = ['oneshot', 'simple', 'replay']

    if not any(x in compatible_modes for x in self._evaluate_mode):
      raise ValueError('Evaluate mode incompatible with dataset: ' + str(self._evaluate_mode))

    if self._evaluate_mode[0] in ['oneshot', 'replay']:
      self._exp_mode = 'oneshot'
      self._train_size = 400
      self._test_size = 400

    elif self._evaluate_mode[0] == 'simple':
      self._exp_mode = 'supervised'
      self._run_num = self._evaluate_mode[1]
      self._train_size = 8464
      self._test_size = 2116

    self._train_files = []
    self._train_labels = []
    self._test_files = []
    self._test_labels = []

  def set_shape(self, height, width):
    self._dataset_shape[1] = height
    self._dataset_shape[2] = width

  def get_train(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot training data.

    Return [random batch_size chars from alphabet_i, random batch_size chars from alphabet_i+1, ....]
    Note, exact same chars and alphabets as returned for test.
    """

    if not self._train_files:
      self._create_datasets()
    return self._dataset(self._train_files, self._train_labels)

  def get_test(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot test data.

    Return [random batch_size chars from alphabet_i, random batch_size chars from alphabet_i+1, ....]
    Note, exact same chars and alphabets as returned for train.

    """
    if not self._test_files:
      self._create_datasets()
    return self._dataset(self._test_files, self._test_labels)

  def _create_datasets(self):
    """
    tf.data.Dataset object for Omniglot test data.

    The order of samples is such that they are divided into batches,
    and always have one of each of the first `batch_size` classes from `test_show_classes`
    """

    self._get_eval_classes()

    # Parameters
    if self._exp_mode == 'oneshot':
      num_runs = 20  # number of classification runs
    else:
      num_runs = 1

    images_folder = os.path.join(self._directory, self._name, 'all_runs_lakelike')

    self._train_files = []
    self._train_labels = []
    self._test_files = []
    self._test_labels = []

    for r in range(1, num_runs + 1):
      if self._run_num:
        run_folder = self._run_num
      else:
        run_folder = 'run' + str(r).zfill(2)

      run_path = os.path.join(images_folder, run_folder, self._exp_mode)

      test_files, test_labels, train_files, train_labels = self._data_run_folder(run_path)

      self._train_files.extend(train_files)
      self._test_files.extend(test_files)
      self._train_labels.extend(train_labels)
      self._test_labels.extend(test_labels)

  def _data_run_folder(self, folder):
    """
    Return train and test files and labels for ONE given run, specified in `folder`
    """

    def extract_labels(files):
      labels = []
      for file in files:
        file, _ = os.path.splitext(file)

        if self._exp_mode == 'oneshot':
          label = int(file.split('_')[1])
        elif self._exp_mode == 'supervised':
          label = int(file.split('_')[0])

        labels.append(label)
      labels = np.array(labels)

      label_idx_arr = np.zeros_like(labels, dtype=np.int32)

      for i, label in enumerate(labels):
        label_idx_arr[i] = self.eval_classes.index(label)

      return label_idx_arr

    train_folder_path = os.path.join(folder, 'training')
    test_folder_path = os.path.join(folder, 'test')

    train_files = os.listdir(train_folder_path)
    train_files.sort()

    train_labels = extract_labels(train_files)

    test_files = os.listdir(test_folder_path)
    test_files.sort()

    test_labels = extract_labels(test_files)

    test_files = [folder + '/test/' + file for file in test_files]
    train_files = [folder + '/training/' + file for file in train_files]

    return test_files, test_labels, train_files, train_labels

  def _dataset(self, filenames, labels):  # pylint: disable=arguments-differ
    """Return dataset object of this list of filenames / labels"""

    def parse_function(filename, label):
      return OmniglotLakeDataset.parse_function(filename, label, self.shape, self._dataset_shape, centre=True)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    return dataset

  def _get_eval_classes(self):
    """Get the image filename and label for each Omniglot character."""
    eval_folder = os.path.join(self._directory, self.name, 'images_evaluation')
    _, eval_labels = super()._filenames_and_labels(eval_folder)

    print(eval_labels[:20])
    self.eval_classes = list(np.unique(eval_labels))
