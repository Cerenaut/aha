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

"""DGSAE class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pagi.components.sparse_autoencoder_component import SparseAutoencoderComponent
from pagi.utils.tf_utils import tf_build_top_k_mask_op


class DGSAE(SparseAutoencoderComponent):
  """Dentate Gyrus (DG) based on the Sparse Autoencoder."""

  @staticmethod
  def default_hparams():
    hparams = SparseAutoencoderComponent.default_hparams()
    hparams.add_hparam('inhibition_decay', 0.8)
    hparams.add_hparam('knockout_rate', 0.25)
    hparams.add_hparam('init_scale', 10.0)

    return hparams

  def get_encoding_op(self):
    return self._dual.get_op('encoding')

  def get_encoding(self):
    return self._dual.get_values('encoding')

  def _build(self):
    super()._build()

    # Override encoding to become binary mask
    encoding = super().get_encoding_op()
    sample_volume = np.prod(encoding.get_shape().as_list()[1:])
    top_k_mask = tf_build_top_k_mask_op(input_tensor=encoding,
                                        k=self._hparams.sparsity,
                                        batch_size=self._hparams.batch_size,
                                        input_area=sample_volume)
    self._dual.set_op('encoding', top_k_mask, override=True)

  def _build_kernel_init(self, input_area, hidden_size):
    """Custom initialization *does* make a big difference to orthogonality, even with inhibition"""
    # Things I've tried:
    # kernel_initializer = None  # default
    # kernel_initializer = tf.initializers.orthogonal(gain=10.0)
    # kernel_initializer = tf.initializers.uniform_unit_scaling(factor=1.0)
    num_weights = input_area * hidden_size
    random_values = np.random.rand(num_weights)
    #random_values = random_values * 2.0 - 1.0
    knockout_rate = self._hparams.knockout_rate
    keep_rate = 1.0 - knockout_rate
    initial_mask = np.random.choice([0, 1], size=(num_weights), p=[knockout_rate, keep_rate])
    initial_values = random_values * initial_mask * self._hparams.init_scale

    for i in range(0, hidden_size):
      w_sum = 0.0
      for j in range(0, input_area):
        #offset = i * input_area + j
        offset = j * hidden_size + i
        w_ij = initial_values[offset]
        w_sum = w_sum + abs(w_ij)
      w_norm = 1.0 / w_sum
      for j in range(0, input_area):
        #offset = i * input_area + j
        offset = j * hidden_size + i
        w_ij = initial_values[offset]
        w_ij = w_ij * w_norm
        initial_values[offset] = w_ij

    kernel_initializer = tf.constant_initializer(initial_values)
    return kernel_initializer

  def _build_filtering(self, training_encoding, testing_encoding):
    """Build the encoding filtering."""
    top_k_input = training_encoding
    top_k2_input = testing_encoding
    hidden_size = self._hparams.filters
    batch_size = self._hparams.batch_size
    k = int(self._hparams.sparsity)
    inhibition_decay = self._hparams.inhibition_decay

    cells_shape = [hidden_size]
    batch_cells_shape = [batch_size, hidden_size]
    inhibition = tf.zeros(cells_shape)
    filtered = tf.constant(np.zeros(batch_cells_shape), dtype=tf.float32)
    training_filtered = filtered

    # Inhibit over time within a batch (because we don't bother having repeats for this).
    for i in range(0, batch_size):

      # Create a mask with a 1 for this batch only
      this_batch_mask_np = np.zeros([batch_size,1])
      this_batch_mask_np[i][0] = 1.0
      this_batch_mask = tf.constant(this_batch_mask_np, dtype=tf.float32)

      refraction = 1.0 - inhibition
      refraction_2d = tf.expand_dims(refraction, 0)  # add batch dim
      refracted = tf.abs(top_k_input) * refraction_2d

      # Find the "winners". The top k elements in each batch sample. this is
      # what top_k does.
      # ---------------------------------------------------------------------
      top_k_mask = tf_build_top_k_mask_op(refracted, k, batch_size, hidden_size)

      # Retrospectively add batch-sparsity per cell: pick the top-k (for now
      # k=1 only). TODO make this allow top N per batch.
      # ---------------------------------------------------------------------
      batch_filtered = training_encoding * top_k_mask  # apply mask 3 to output 2
      this_batch_filtered = batch_filtered * this_batch_mask

      this_batch_topk = top_k_mask * this_batch_mask
      fired = tf.reduce_max(this_batch_topk, axis=0)  # reduce over batch

      inhibition = inhibition * inhibition_decay + fired  # set to 1

      training_filtered = training_filtered + this_batch_filtered

    testing_filtered = training_filtered
    return training_filtered, testing_filtered
