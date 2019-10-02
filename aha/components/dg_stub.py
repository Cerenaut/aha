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

"""DGStubComponent class."""

import numpy as np
import tensorflow as tf

from pagi.utils.dual import DualData
from pagi.components.summary_component import SummaryComponent
from pagi.components.summarize_levels import SummarizeLevels


class DGStubComponent(SummaryComponent):
  """Dentate Gyrus (DG) Stub Component to use in the Episodic Component."""

  @staticmethod
  def default_hparams():
    return tf.contrib.training.HParams(
        batch_size=20,
        filters=50,
        sparsity=5,
        summarize_levels=SummarizeLevels.ALL.value,   # does nothing
        max_outputs=3                                 # does nothing
    )

  def __init__(self):
    self._name = None
    self._hidden_name = None
    self._hparams = None
    self._dual = None

  @property
  def name(self):
    return self._name

  def reset(self):
    pass

  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Add items to the feed dict to run a batch."""

  def build_summaries(self, batch_types=None, max_outputs=3, scope=None):
    """Build any summaries needed to debug the module."""

  def _build_summaries(self, batch_type, max_outputs=3):
    """Build summaries for this batch type. Can be same for all batch types."""

  def write_summaries(self, step, writer, batch_type='training'):
    """Write any summaries needed to debug the module."""

  def add_fetches(self, fetches, batch_type='training'):
    """Add graph ops to the fetches dict so they are evaluated."""
    names = ['encoding']
    self._dual.add_fetches(fetches, names)

  def set_fetches(self, fetched, batch_type='training'):
    """Store results of graph ops in the fetched dict so they are available as needed."""
    names = ['encoding']
    self._dual.set_fetches(fetched, names)

  def _dg_stub_batch(self):
    """
    Return batch of non overlapping n-hot samples in range [0,1]
    All off-graph
    """

    batch_size = self._hparams.batch_size
    sample_size = self._hparams.filters
    n = self._hparams.sparsity

    assert ((batch_size * n - 1) + n) < sample_size, "Can't produce batch_size {0} non-overlapping samples, " \
           "reduce n {1} or increase sample_size {2}".format(batch_size, n, sample_size)

    batch = np.zeros(shape=(batch_size, sample_size))

    # return the sample at given idx
    for idx in range(batch_size):
      start_idx = idx * n
      end_idx = start_idx + n
      batch[idx][start_idx:end_idx] = 1

    return batch

  def get_encoding_op(self):
    return self._dual.get_op('encoding')

  def get_encoding(self):
    return self._dual.get_values('encoding')

  def get_decoding(self):
    return self._dual.get_values('encoding')    # return encoding, same thing for both

  def get_loss(self):
    return 0.0

  def build(self, hparams, name='dg_stub'):
    """Builds the DG Stub."""
    self._name = name
    self._hparams = hparams
    self._dual = DualData(self._name)

    batch_arr = self._dg_stub_batch()

    the_one_batch = tf.convert_to_tensor(batch_arr, dtype=tf.float32)
    self._dual.set_op('encoding', the_one_batch)

    # add a stub of secondary decoding also, it is expected by workflow
    self._dual.add('secondary_decoding_input', shape=the_one_batch.shape, default_value=1.0).add_pl(default=True)
