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

"""DGSCAE class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pagi.components.sparse_conv_autoencoder_component import SparseConvAutoencoderComponent
from pagi.utils.tf_utils import tf_build_top_k_mask_op


class DGSCAE(SparseConvAutoencoderComponent):
  """Dentate Gyrus (DG) based on the Sparse Convolutional Autoencoder."""

  @staticmethod
  def default_hparams():
    hparams = SparseConvAutoencoderComponent.default_hparams()
    hparams.add_hparam('flat_sparsity', 5)
    return hparams

  def get_encoding_op(self):
    return self._dual.get_op('encoding')

  def get_encoding(self):
    return self._dual.get_values('encoding')

  def _build(self):
    super()._build()

    encoding = super().get_encoding_op()
    sample_volume = np.prod(encoding.get_shape().as_list()[1:])
    top_k_mask = tf_build_top_k_mask_op(input_tensor=encoding,
                                        k=self._hparams.flat_sparsity,
                                        batch_size=self._hparams.batch_size,
                                        input_area=sample_volume)

    self._dual.set_op('encoding', top_k_mask, override=True)
