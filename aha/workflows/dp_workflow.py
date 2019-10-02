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

"""Workflow base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import tensorflow as tf
import numpy as np
import random

from pagi.utils import image_utils
from pagi.workflows.workflow import Workflow


class MockData:
  def __init__(self):
    self.shape = None


class DPWorkflow(Workflow):
  """Workflow for testing diff plasticity on a mock dataset."""

  @staticmethod
  def generate_input_bands(sample_size, number_samples, sparsity=0.5,
                           min_value=-1):

    episode = np.zeros((number_samples, sample_size))
    episode[0:number_samples, 0:sample_size] = min_value

    # 3 strips of contiguous high values

    strip_width = int((sample_size*(1-sparsity)) // 3)
    range_width = int(sample_size // 3)   # within which a strip of 1's appear

    range1 = range_width
    range2 = range_width*2
    range3 = sample_size

    for i in range(number_samples):

      r1 = np.random.randint(0, range1 - strip_width - 1)
      r2 = np.random.randint(range1, range2 - strip_width - 1)
      r3 = np.random.randint(range2, range3 - strip_width - 1)

      episode[i][r1:r1 + strip_width] = 1
      episode[i][r2:r2 + strip_width] = 1
      episode[i][r3:r3 + strip_width] = 1

    return episode

  @staticmethod
  def generate_random_input(sample_size, number_samples, presentation_repeats, sample_repeats,
                            degraded_repeats, blank_repeats, sparsity, degrade_factor, add_bias,
                            amplify_factor=20, degrade_value=0, min_value=-1, binary=True):
    """
    Generate the full list of inputs for an episode. The final item is a degraded copy of a randomly chosen item.

    This is intentionally copied from the Uber paper for comparison (with variable names modified).
    So there are some specifics that look strange out of context.

    :param sample_size: size of each item in the episode
    :param number_samples: The number of patterns to learn in each episode
    :param presentation_repeats: Number of times each pattern is to be presented
    :param sample_repeats: Number of time steps for each presentation
    :param degraded_repeats: Same thing but for the final test pattern
    :param blank_repeats: Duration of zero-input interval between presentations
    :param sparsity: fractional sparsity e.g. 0.5 = 0.5 active,   0.2 = 0.8 active
    :param degrade_factor: Proportion of bits to zero out in the target pattern at test time
    :param add_bias:
    :param amplify_factor:
    :param degrade_value:
    :param min_value:
    :param binary: generate binary values (0,1) or if false, floats between [0 and 1)
    :return: target (non degraded item), numpy array of shape = [episode_length, filters]
    """

    filters = sample_size
    if add_bias:
      filters = filters + 1

    episode_length = DPWorkflow.calc_generate_inputs_length(number_samples=number_samples,
                                                            presentation_repeats=presentation_repeats,
                                                            sample_repeats=sample_repeats,
                                                            degraded_repeats=degraded_repeats,
                                                            blank_repeats=blank_repeats)

    inputT = np.zeros((episode_length, filters))

    # Create the random patterns to be memorized in an episode
    length_sparse = int(sample_size * sparsity)

    if binary:
      seedp = np.ones(sample_size)
    else:
      seedp = np.random.rand(sample_size)

    seedp[:length_sparse] = min_value
    patterns = []
    for nump in range(number_samples):
      p = np.random.permutation(seedp)
      patterns.append(p)

    # Now 'patterns' contains the episode_length patterns to be memorized in this episode - in numpy format
    # Choosing the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    preservedbits = np.ones(sample_size)
    preservedbits[:int(degrade_factor * sample_size)] = degrade_value
    np.random.shuffle(preservedbits)
    degradedtestpattern = testpattern * preservedbits

    logging.debug("test pattern     = ", testpattern)
    logging.debug("degraded pattern = ", degradedtestpattern)

    # Inserting the inputs in the input tensor at the proper places
    for nc in range(presentation_repeats):
      np.random.shuffle(patterns)
      for ii in range(number_samples):
        for nn in range(sample_repeats):
          numi = nc * (number_samples * (sample_repeats + blank_repeats)) + ii * (sample_repeats + blank_repeats) + nn
          inputT[numi][:sample_size] = patterns[ii][:]

    # Inserting the degraded pattern
    for nn in range(degraded_repeats):
      logging.debug("insert degraded pattern at: [{0},{1},:{2}]".format(-degraded_repeats + nn, 0, sample_size))
      inputT[-degraded_repeats + nn][:sample_size] = degradedtestpattern[:]

    for nn in range(episode_length):
      if add_bias:
        inputT[nn][-1] = 1.0  # Bias neuron
      inputT[nn] *= amplify_factor  # Strengthen inputs

    logging.debug("shape of inputT: ", np.shape(inputT))

    return inputT, testpattern

  @staticmethod
  def calc_generate_inputs_length(number_samples, presentation_repeats, sample_repeats,
                                  degraded_repeats, blank_repeats):
    episode_length = presentation_repeats * (
            (sample_repeats + blank_repeats) * number_samples) + degraded_repeats  # Total number of steps per episode
    return episode_length

  def _setup_dataset(self):

    episode_size = self._opts['episode_size']
    batch_length = DPWorkflow.calc_generate_inputs_length(number_samples=episode_size,
                                                          presentation_repeats=1,
                                                          sample_repeats=1,
                                                          degraded_repeats=0,
                                                          blank_repeats=0)

    sample_size = self._hparams.filters
    bias_neurons = self._hparams.bias_neurons

    batch_pl = tf.placeholder(tf.float32, [batch_length, sample_size], name='dataset_handle')
    self._placeholders['dataset_handle'] = batch_pl
    self._inputs = batch_pl

    # 'inherent' shape of the data, this will be used for visualisation
    self._dataset = MockData()
    self._dataset.shape, _ = image_utils.square_image_shape_from_1d(sample_size + bias_neurons)

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""

    episode_size = self._opts['episode_size']  # generate this many samples in an episode
    sparsity = self._opts['sparsity']  # generate data with this sparsity (0.8 = 0.2 are active)
    min_value = self._opts['min_value']  # generate data with non-active bits at this value
    random_pattern = self._opts['random_pattern']  # generate random pattern, or one with structure (true/false)
    bias_neuron = self._opts['bias_neuron']  # generate data with one bit always active

    sample_size = self._hparams.filters   # tied to the diff plasticity component, must be same as number filters

    self._on_before_training_batches()

    for batch_num in range(self._last_step, num_batches):
      if random_pattern:

        sample_size_with_bias_neuron = sample_size
        if bias_neuron:
          sample_size_with_bias_neuron = sample_size - 1
        episode, _ = DPWorkflow.generate_random_input(sample_size=sample_size_with_bias_neuron,
                                                      number_samples=episode_size,
                                                      presentation_repeats=1,
                                                      sample_repeats=1,
                                                      degraded_repeats=0,
                                                      blank_repeats=0,
                                                      sparsity=sparsity,
                                                      degrade_factor=0,
                                                      add_bias=bias_neuron,
                                                      amplify_factor=1,
                                                      degrade_value=0,
                                                      min_value=min_value,
                                                      binary=True)
      else:
        episode = DPWorkflow.generate_input_bands(sample_size=sample_size,
                                                  number_samples=episode_size,
                                                  sparsity=sparsity,
                                                  min_value=min_value)

      training_step = self._session.run(tf.train.get_global_step(self._session.graph))
      training_epoch = training_step
      # training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)

      # Perform the training, and retrieve feed_dict for evaluation phase
      self.training(episode, batch_num)

      self._on_after_training_batch(batch_num, training_step, training_epoch)

      # Export any experiment-related data
      # -------------------------------------------------------------------------
      if self._export_opts['export_filters']:
        if (batch_num + 1) % self._export_opts['interval_batches'] == 0:
          self.export(self._session)

      if self._export_opts['export_checkpoint']:
        if (batch_num + 1) % num_batches == 0:
          self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch_num + 1)

