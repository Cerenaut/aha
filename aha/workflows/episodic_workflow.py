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

"""EpisodicWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pagi.utils import layer_utils
from pagi.workflows.composite_workflow import CompositeWorkflow

from aha.components.episodic_component import EpisodicComponent


class EpisodicWorkflow(CompositeWorkflow):
  """Enables usage of the EpisodicComponent with the standard Workflow."""

  ################################################
  # Overrides

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        train_classes=[5, 6, 7, 8, 9],
        test_classes=[5, 6, 7, 8, 9],
        degrade_type='vertical',  # vertical, horizontal or random: the model completes image degraded by this method
        degrade_step='hidden',  # 'test', 'input', 'hidden' or 'none'
        evaluate=True,
        train=True,
        visualise_vc=False,
        visualise_dg_at_vc=False,
        visualise_pc_at_dg=False,
        visualise_pc_at_vc=False
    )

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):
    super().__init__(session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
                     opts, summarize, seed, summary_dir, checkpoint_opts)
    self._train_loss_pr = 0
    self._train_loss_pm = 0
    self._train_loss_pm_raw = 0
    self._logging_freq = 10

  def _test_consistency(self):
    """
    If multiple params relate to each other, make sure they are set consistently
    - prevent hard to diagnose runtime issues
    """

    if self._opts['evaluate'] and self._opts['test_recurse']:
      if self._hparams.pc_type != 'hl':
        raise RuntimeError("There can be no test recursion unless using Hopfield fo PC.\n"
                           "Set the workflow option `test_recurse` to false")

    if self._opts['train'] and self._opts['train_recurse']:
      if self._hparams.pc_type != 'hl':
        raise RuntimeError("No train recursion unless using Hopfield for PC.\n"
                           "Set the workflow option `train_recurse` to false")
      else:
        if self._hparams.pc_hl_memorise_method == 'pinv':
          raise RuntimeError("There can be no train recursion when using Hopfield if memorisation method is 'pinv'.\n"
                             "Set the workflow option `train_recurse` to false or change memorise method.")

    if self._hparams.pc_type == 'hl':
      if self._opts['evaluate'] and not self._opts['test_recurse']:
          raise RuntimeWarning("Using Hopfield for evaluation, are you sure you want 'test_recurse == False'")

      if self._opts['train'] and not self._opts['train_recurse'] and \
              not self._hparams.pc_hl_memorise_method == 'pinv':
          raise RuntimeWarning("Using Hopfield for train, without pinv method, "
                               "are you sure you want train_recurse == False'")

  def _setup_component(self):
    """Setup the component"""

    self._component = self._component_type()
    self._component._degrade_type = self._opts['degrade_type']

    labels = self._labels

    if labels.dtype != tf.string:
      self._labels = tf.Print(self._labels, [self._labels], summarize=self._hparams.batch_size)
      labels = tf.one_hot(self._labels, self._dataset.num_classes)

    self._component.build(self._inputs, self._dataset.shape, self._hparams,
                         label_values=labels, name='episodic')

    if self._summarize:
      batch_types = ['training', 'encoding']
      if self._freeze_training:
        batch_types.remove('training')
      self._component.build_summaries(batch_types)  # Ask the component to unpack for you

  def _on_before_training_batches(self):
    print('\n\n')
    logging.info('| Training Step | Loss - vc     | Loss - dg     | Loss - pc     | Batch    |   Epoch  |')
    logging.info('|---------------|---------------|---------------|---------------|----------|----------|')

  def _on_after_training_batch(self, batch, training_step, training_epoch):
    if (batch + 1) % self._logging_freq == 0:
      # Explore the output of the batch
      loss_vc = self._component.get_vc().get_loss()

      loss_dg = 0
      if self._component.get_dg() is not None:
        loss_dg = self._component.get_dg().get_loss()

      loss_pc = 0
      if self._component.get_pc() is not None:
        pc = self._component.get_pc()
        loss_pc = pc.get_loss()

        if pc.use_input_cue:
          self._train_loss_pr = pc.get_loss_pr(0)
        if pc.use_pm or pc.use_pm_raw:
          self._train_loss_pm, self._train_loss_pm_raw = pc.get_losses_pm(0)

      output = '| {0:>13} | {1:13.4f} {2:13.4f} ({3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}) | Batch {7}  | Epoch {8}  |'.format(
          training_step, loss_vc, loss_dg, loss_pc, self._train_loss_pr,
          self._train_loss_pm, self._train_loss_pm_raw,
          batch, training_epoch)

      logging.info(output)

  def export(self, session, feed_dict):
    # Export all filters to disk
    self._component.write_filters(session, self._summary_dir)

    visualize_filters = False

    if visualize_filters:
      vc_encoding = self._component.get_vc().get_encoding()
      self._decode_and_write_vc_filters(
          shape=list(vc_encoding.shape),
          feed_dict=feed_dict)

  def _decode_and_write_vc_filters(self, shape, feed_dict):
    """Generates VC encoding-shaped data where all values are zero,
    except for the filter being visualised (z). These values are then decoded
    using the VC and saved to disk."""

    vc_num_filters = shape[3]

    # Create input data such that filter Z is active (=1, rest zeros)
    vc_filters = []
    for z in range(vc_num_filters):
      vc_filter = np.zeros(shape[1:])
      x = shape[1] // 2
      y = shape[2] // 2
      vc_filter[x, y, z] = 1
      vc_filters.append(vc_filter)

    # Must split into N=batch_size chunks due to static batch size
    vc_filters_chunks = [
        vc_filters[x:x+self._hparams.batch_size] for x in range(
            0, len(vc_filters), self._hparams.batch_size)
    ]

    logging.info('VC Filters: %s, Number of chunks: %s',
                 np.array(vc_filters).shape,
                 len(vc_filters_chunks))

    vc_decoded_filters = []
    for vc_filters_chunk in vc_filters_chunks:
      if len(vc_filters_chunk) != self._hparams.batch_size:
        logging.warning('Skipping filter chunk due to even/odd mismatch '
                        'between the batch size and number of filters.')
        continue

      vc_decoded_filter = self._decoder(0, 'vc_filters', 'vc',
                                        vc_filters_chunk, feed_dict,
                                        summarise=False)
      vc_decoded_filters.append(vc_decoded_filter)

    if vc_decoded_filters:
      vc_decoded_filters = np.array(vc_decoded_filters)     # [ #batch-chunks, batch size, width, height, channels]

      vc_decoded_filters_flat = vc_decoded_filters.reshape(               # [filters, width, height, channels]
          vc_decoded_filters.shape[0] * vc_decoded_filters.shape[1],
          vc_decoded_filters.shape[2], vc_decoded_filters.shape[3],
          vc_decoded_filters.shape[4])

      for idx, decoded_filter in enumerate(vc_decoded_filters_flat):
        decoded_filter = decoded_filter.reshape(decoded_filter.shape[0], decoded_filter.shape[1])
        filetype = 'png'
        filename = 'decoded_filter_' + str(idx) + '.' + filetype
        filepath = os.path.join(self._summary_dir, filename)

        if not os.path.isfile(filepath):
          plt.title('VC Filter ' + str(idx))
          plt.imshow(decoded_filter, interpolation='none')
          plt.savefig(filepath, dpi=300, format=filetype)
          plt.close()

  ################################################
  # episodic component can decode from sub components, these can be overridden

  def _is_using_visualise_vc(self):
    return self._opts['visualise_vc']

  def _is_using_visualise_pc_at_dg(self):
    return self._hparams.pc_type != 'none' and self._hparams.dg_type != 'none' and self._opts['visualise_pc_at_dg']

  def _is_using_visualise_pc_at_vc(self):
    return self._hparams.pc_type != 'none' and self._opts['visualise_pc_at_vc']

  def _is_using_visualise_dg_at_vc(self):
    return self._hparams.dg_type != 'none' and self._opts['visualise_dg_at_vc']

  def _is_using_if_at_vc(self):
    return self._hparams.use_interest_filter and self._opts['visualise_if_at_vc']

  def _is_decoding_vc_at_vc(self):
    return self._is_using_visualise_vc()

  def _is_decoding_pc_at_dg(self):
    return self._is_using_visualise_pc_at_dg()

  def _is_decoding_pc_at_vc(self):
    return self._is_using_visualise_pc_at_vc()

  def _is_decoding_dg_at_vc(self):
    return self._is_using_visualise_dg_at_vc()

  def _is_decoding_if_at_vc(self):
    return self._is_using_if_at_vc()

  ################################################
  # Useful operations for episodic component

  def _setup_train_feed_dict(self, batch_type, training_handle):
    """
    Add batch_type placeholder values in feed_dict
    --> sae now needs batch type through a pl for Filtering (to be different for training and encoding)
    """

    feed_dict_batch_type = {}
    component_batch_type_pl = self._component.get_batch_type()  # dic -> (component, batch type)

    for c in component_batch_type_pl:
      feed_dict_batch_type[component_batch_type_pl[c]] = batch_type['episodic/' + c]  # dict->(episodic/'c', batch_type)

    feed_dict = {
        self._placeholders['dataset_handle']: training_handle
    }

    feed_dict.update(feed_dict_batch_type)

    return feed_dict

  def _set_degrade_options(self, feed_dict):
    """
    Updates the feed_dict to determine degradation (hidden, input or none) in later executions.
    Makes a copy of the feed_dict first, and returns updated copy.
    """
    testing_feed_dict = feed_dict.copy()  # Avoid mutating the original feed_dict

    # Get the degrade placeholder
    degrade_step_pl = self._component.get_dual().get('degrade_step').get_pl()
    degrade_random_pl = self._component.get_dual().get('degrade_random').get_pl()

    testing_feed_dict.update({
        degrade_step_pl: self._opts['degrade_step'],  # test, input, hidden or none
        degrade_random_pl: random.uniform(0, 1)
    })

    return testing_feed_dict
