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

"""RecursiveComponentHarness class."""

import logging


class RecursiveComponentHarness(object):
  """
  A Harness to run the recursive sparse autoencoder and other recursive components in the future.
  The recursive component requires multiple 'inference' steps before a training step.
  This class encapsulates that functionality.
  """

  def __init__(self, component, recurse_iterations=1):
    """Initializes the harness parameters."""
    self._component = component
    self._recurse_iterations = recurse_iterations
    self._summarize_recursive = True
    self._info_freq = 10

  def _set_input_gain_pl(self, feed_dict, gain):
    """Set the input gain placeholder for the next graph step"""
    self._component.update_feed_dict_input_gain_pl(feed_dict, gain)

  def _set_input_partial_pl(self, feed_dict):
    """Set the partial mask placeholder"""
    self._component.set_input_partial_pl(feed_dict)

  def _reset_partial_mask(self):
    """A mask is used to degrade the input to be a partial pattern. This creates a new random mask"""

    if hasattr(self._component, 'reset_partial_mask'):
      self._component.reset_partial_mask(0.5)

  def _pre(self, feed_dict, batch_type, fetches=None):
    """Setup feed_dict and fetches, before a step"""

    self._component.update_feed_dict(feed_dict, batch_type)
    if fetches is None:
      fetches = {}
    self._component.add_fetches(fetches, batch_type)
    return fetches

  def _post(self, fetched, batch_type):
    self._component.set_fetches(fetched, batch_type)

  def step(self, session, writer, summary_batch, feed_dict, mode, input_mode,
           inference_batch_type,
           train_batch_type=None,
           fetches=None):
    """

    :param session:
    :param writer: write recursive updates to this summary, if not None
    :param summary_batch:
    :param feed_dict: feed_dict with necessary placeholders
    :param mode: 'inference' or 'training'. In inference mode, no training takes place.
    :param input_mode:
    :param train_batch_type: batch type component uses training (required if in 'train' mode, otherwise it is ignored)
    :param inference_batch_type: the batch type component uses when being used for inference
    :param fetches:
    :return: most recently fetched ops
    """

    if fetches is None:
      fetches = {}

    summarise = self._summarize_recursive and summary_batch >= 0

    logging.debug("\t\t---- Summary Batch: %s", str(summary_batch))
    logging.debug("\t\t---- --> %s", ("Write summaries" if summarise is not None else "do NOT write SUMMARY"))

    # important to reset, so that initial feedback value is zero (reset hidden values to zero)
    self._component.reset()

    self._reset_partial_mask()    # present different partial pattern every batch

    feed_dict_inference = feed_dict.copy()
    feed_dict_train = feed_dict.copy()

    if mode == 'training':
      inference_iterations = self._recurse_iterations - 1
    else:
      inference_iterations = self._recurse_iterations

    # run a 'recursive set' i.e. 'iteration' iterations without training the network
    for i in range(inference_iterations):
      step = summary_batch * self._recurse_iterations + i
      logging.debug("\t\t-- Inference step: %s", str(step))

      self._manage_input(i, mode, input_mode, feed_dict_inference)

      fetches = self._pre(feed_dict_inference, inference_batch_type, fetches)
      fetched = session.run(fetches, feed_dict=feed_dict_inference)
      self._post(fetched, inference_batch_type)

      logging.debug('\t\t-- labels: {}'.format(fetched['labels'][0]))

      # write explicit summary for each recursive iteration
      # print('\t\t-- Inference step: ', str(step), ' write summaries? ', str(summarise))
      summarise = True  # Otherwise this doesn't fire at recursion time.
      if summarise:
        self._component.write_recursive_summaries(step, writer, inference_batch_type)

      # if i % self._info_freq == 0 or i == inference_iterations - 1:
        # logging.info("Recursive iteration {0}".format(step))  # +ve values are written to recursive summary

    # run one training step (if in training mode only)
    if mode == 'training':
      step = summary_batch * self._recurse_iterations + inference_iterations
      logging.debug("\t\t-- Training Step: %s", str(step))    # +ve values are written to recursive summary

      self._manage_input(inference_iterations, mode, input_mode, feed_dict_train)

      fetches = self._pre(feed_dict_train, train_batch_type, fetches)
      fetched = session.run(fetches, feed_dict=feed_dict_train)
      self._post(fetched, train_batch_type)

      # write explicit summary for each recursive iteration
      if summarise:
        self._component.write_recursive_summaries(step, writer, train_batch_type)

    return fetched

  def _manage_input(self, i, mode, input_mode, feed_dict):
    """Partially degrade input or completely remove it depending on specified mode."""
    if i == 0:
      # if input should be partial on first presentation (inference or training mode)
      if (mode == 'training' and input_mode['train_first'] == 'partial') or \
              (mode == 'inference' and input_mode['test_first'] == 'partial'):
        self._set_input_partial_pl(feed_dict)

      # if input should be partial on first presentation (inference or training mode)
      if (mode == 'training' and input_mode['train_first'] == 'absent') or \
              (mode == 'inference' and input_mode['test_first'] == 'absent'):
        self._set_input_gain_pl(feed_dict, 0.0)

    if i > 0:
      # if input should be absent on inference loops (inference or training mode)
      if (mode == 'training' and input_mode['train_inference'] == 'absent') or \
              (mode == 'inference' and input_mode['test_inference'] == 'absent'):
        self._set_input_gain_pl(feed_dict, 0.0)

      # if input should be partial on inference loops (inference or training mode)
      if (mode == 'training' and input_mode['train_inference'] == 'partial') or \
              (mode == 'inference' and input_mode['test_inference'] == 'partial'):
        self._set_input_partial_pl(feed_dict)
