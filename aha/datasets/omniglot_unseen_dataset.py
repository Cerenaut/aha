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

"""OmniglotUnseenDataset class."""

import os

import numpy as np

from pagi.datasets.omniglot_dataset import OmniglotDataset


class OmniglotUnseenDataset(OmniglotDataset):
  """Omniglot Unseen Dataset based on tf.data."""

  def __init__(self, directory):
    super(OmniglotUnseenDataset, self).__init__(directory=directory)

    self._train_size = 2600
    self._test_size = 2600
    self._num_train_classes = 659
    self._num_test_classes = 659
    self._num_classes = 659

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for Omniglot training data."""
    return self._dataset(self._directory, 'images_evaluation_unseen')

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for Omniglot test data."""
    return self._dataset(self._directory, 'images_evaluation_unseen')

  def _download(self, directory, filename):
    """Download (and unzip) a file from the Omniglot dataset if not already done."""
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)

    return filepath

  def _filenames_and_labels(self, image_folder):
    """Get the image filename and label for each Omniglot character."""
    eval_folder = os.path.join(self._directory, self.name, 'images_evaluation')
    _, eval_labels = super()._filenames_and_labels(eval_folder)

    self.eval_classes = list(np.unique(eval_labels))

    filename_arr, label_arr = super()._filenames_and_labels(image_folder)

    label_idx_arr = np.zeros_like(label_arr, dtype=np.int32)
    for i, label in enumerate(label_arr):
      label_idx_arr[i] = self.eval_classes.index(label)

    # Since this dataset is a subset of images_evaluation, the labels extracted from filenames
    # won't be one-hot encodable (as its missing the full set of classes).
    # We use the label indices as a pseudo label in order to one-hot the labels for learning.

    return filename_arr, label_idx_arr
