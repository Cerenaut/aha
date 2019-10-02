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

"""OmniglotLakeRunsDataset class.."""

import os
import zipfile
import tempfile
import logging
import copy

from six.moves import urllib

import tensorflow as tf

from pagi.datasets.dataset import Dataset
from pagi.utils.tf_utils import tf_invert_values, tf_centre_of_mass

from aha.datasets.omniglot_lake_dataset import OmniglotLakeDataset


class OmniglotLakeRunsDataset(Dataset):
  """Omniglot Dataset based on tf.data."""

  def __init__(self, directory, batch_size):
    super(OmniglotLakeRunsDataset, self).__init__(
        name='omniglot',
        directory=directory,
        dataset_shape=[-1, 105, 105, 1],
        train_size=400,
        test_size=400,
        num_train_classes=400,
        num_test_classes=400,
        num_classes=400)

    self._batch_size = batch_size

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

    if len(self._train_files) == 0:
      self._create_datasets()
    return self._dataset(self._train_files, self._train_labels)

  def get_test(self, preprocess=False):
    """
    tf.data.Dataset object for Omniglot test data.

    Return [random batch_size chars from alphabet_i, random batch_size chars from alphabet_i+1, ....]
    Note, exact same chars and alphabets as returned for train.

    """
    if len(self._test_files) == 0:
      self._create_datasets()
    return self._dataset(self._test_files, self._test_labels)

  def _download(self, directory, from_filename, to_filename=None):
    """
    Download (and unzip) a file from the Omniglot dataset if not already done.
    `from_filename` is relative to omniglot github python folder
    `to_filename` is relative to folder: datasets_folder/class_name
    """

    if to_filename is None:
      to_filename = from_filename

    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, to_filename)
    if tf.gfile.Exists(filepath):
      return filepath
    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    url = 'https://github.com/brendenlake/omniglot/raw/master/python/' + (from_filename + '.zip')
    _, zipped_filepath = tempfile.mkstemp(suffix='.zip')
    logging.info('Downloading %s to %s', url, zipped_filepath)
    urllib.request.urlretrieve(url, zipped_filepath)

    zip_ref = zipfile.ZipFile(zipped_filepath, 'r')
    zip_ref.extractall(filepath)        # this differs from other omniglot classes, which extract to the folder we want
    zip_ref.close()

    os.remove(zipped_filepath)
    return filepath

  def _create_datasets(self):
    """
    tf.data.Dataset object for Omniglot test data.

    The order of samples is such that they are divided into batches,
    and always have one of each of the first `batch_size` classes from `test_show_classes`
    """

    # Parameters
    num_runs = 20  # number of classification runs

    images_folder = self._download(self._directory, 'one-shot-classification/all_runs', 'all_runs')

    self._train_files = []
    self._train_labels = []
    self._test_files = []
    self._test_labels = []

    for r in range(1, num_runs + 1):
      rs = str(r)
      if len(rs) == 1:
        rs = '0' + rs

      run_folder = 'run' + rs
      run_path = os.path.join(images_folder, run_folder)

      test_files, test_labels, train_files, train_labels = self._data_run_folder(run_path)

      self._train_files.extend(train_files)
      self._test_files.extend(test_files)
      self._train_labels.extend(train_labels)
      self._test_labels.extend(test_labels)

  def _data_run_folder(self, folder):
    """
    Return train and test files and labels for ONE given run, specified in `folder`
    """

    fname_label = 'class_labels.txt'  # where class labels are stored for each run

    # get file names
    with open(folder + '/' + fname_label) as f:
      content = f.read().splitlines()
    pairs = [line.split() for line in content]

    test_files = [pair[0] for pair in pairs[0:self._batch_size]]
    train_files = [pair[1] for pair in pairs[0:self._batch_size]]

    train_labels = copy.copy(train_files)
    test_labels = copy.copy(train_files)      # same labels as train, because we'll read them in this order

    test_files = [folder + '/../' + file for file in test_files]
    train_files = [folder + '/../' + file for file in train_files]

    return test_files, test_labels, train_files, train_labels

  def _dataset(self, filenames, labels):
    """Return dataset object of this list of filenames / labels"""

    def parse_function(filename, label):
      return OmniglotLakeDataset.parse_function(filename, label, self.shape, self._dataset_shape, centre=True)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    return dataset
