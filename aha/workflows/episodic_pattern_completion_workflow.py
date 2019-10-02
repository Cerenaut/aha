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

"""EpisodicPatternCompletionWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pagi.workflows.episodic_workflow import EpisodicWorkflow
from pagi.workflows.pattern_completion_workflow import PatternCompletionWorkflow, UseTrainForTest


class EpisodicPatternCompletionWorkflow(EpisodicWorkflow, PatternCompletionWorkflow):
  """
  Enables usage of the EpisodicComponent with the PatternCompletionWorkflow.
  """

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        num_repeats=1,
        superclass=False,
        class_proportion=1.0,
        invert_images=False,
        min_val=0,  # set any 0 in the input image, to this new min_val. ---> if >0, then don't do anything
        train_classes=['5', '6', '7', '8', '9'],
        test_classes=['5', '6', '7', '8', '9'],
        degrade_type='vertical',  # vertical, horizontal or random: the model completes image degraded by this method
        degrade_step='hidden',    # 'test' (apply at gen of test set), or 'input', 'hidden', 'none' (applied in graph)
        completion_gain=1.0,
        train_recurse=False,
        test_recurse=False,
        recurse_iterations=5,  # if >1, then PC is recursive (only supported for Hopfield i.e. no recursion on training)
        rsummary_batches=2,
        input_mode={
            "train_first": "complete",
            "train_inference": "complete",
            "test_first": "complete",
            "test_inference": "complete"
        },
        evaluate=True,
        train=True,
        visualise_vc=False,
        visualise_dg_at_vc=False,
        visualise_pc_at_dg=False,
        visualise_pc_at_vc=False,
        evaluate_mode='simple'  # simple = calc compl. of pc use pattern_completion_workflow,
                                # expA_isolate_view = test completion and visualise at each stage
                                # expA_isolate = test completion and range of tests to isolate performance of components
    )

  def _test_consistency(self):
    """
    If multiple params relate to each other, make sure they are set consistently
    - prevent hard to diagnose runtime issues
    """
    if (self._opts['visualise_dg_at_vc'] or self._opts['visualise_pc_at_dg']) and \
            self._hparams.dg_type == "none":
      raise RuntimeError("Cannot visualise DG in VC if there is no DG.S\net workflow param visualise_dg_at_vc to False")

  def _is_write_evaluate_summary(self):
    return False

  def _is_decoding_vc_at_vc(self):
    return self._is_using_visualise_vc()

  def _is_decoding_dg_at_vc(self):
    return self._is_using_visualise_dg_at_vc()

  def _is_decoding_pc_at_dg(self):
    return self._is_using_visualise_pc_at_dg()

  def _is_decoding_pc_at_vc(self):
    return self._is_using_visualise_pc_at_vc()

  def _is_decoding_if_at_vc(self):
    return self._is_using_if_at_vc()

  def _generate_datasets(self):
    """Override in child classes with your options"""

    degrade_test = False
    if self._opts['degrade_step'] == 'test':
      degrade_test = True

    use_trainset_for_tests = UseTrainForTest.IDENTICAL    # can be different in few shot workflow

    train_dataset, test_dataset = self._gen_datasets_with_options(self._opts['train_classes'],
                                                                  self._opts['test_classes'],
                                                                  is_superclass=self._opts['superclass'],
                                                                  class_proportion=self._opts['class_proportion'],
                                                                  degrade_test=degrade_test,
                                                                  degrade_type=self._opts['degrade_type'],  # only relevant if degrade_test = True
                                                                  degrade_val=self._opts['min_val'],        # only relevant if degrade_test = True
                                                                  recurse_train=self._is_train_recursive(),
                                                                  recurse_test=self._is_inference_recursive(),
                                                                  num_batch_repeats=self._opts['num_repeats'],
                                                                  recurse_iterations=self._opts['recurse_iterations'],
                                                                  evaluate_step=self._opts['evaluate'],
                                                                  use_trainset_for_tests=use_trainset_for_tests,
                                                                  invert_images=self._opts['invert_images'],
                                                                  min_val=self._opts['min_val'])
    return train_dataset, test_dataset

  def _complete_pattern(self, feed_dict, training_handle, test_handle, test_step):
    if self._opts['evaluate_mode'] == 'simple':
      return super()._complete_pattern(feed_dict, training_handle, test_handle, test_step)
    elif self._opts['evaluate_mode'] == 'expA_isolate':
      return self._complete_pattern_isolate(feed_dict, training_handle, test_handle)
    elif self._opts['evaluate_mode'] == 'expA_isolate_view':
      return self._complete_pattern_isolate_view(feed_dict, training_handle, test_handle, test_step)

  def _setup_recursive_train_modes(self, batch_type):
    """
    Set the appropriate mode depending on batch type.
    This is only called when recursive training
    """
    mode = 'training'
    # If each component is in encoding mode
    if all(v == 'encoding' for v in batch_type.values()):
      mode = 'inference'
    return mode

  def _complete_pattern_isolate_view(self, feed_dict, training_handle, test_handle, test_step):
    """
    Exposes component to an incomplete pattern and calculates the completion loss.
    - Iterates train set ONCE,
    - Iterates test set multiple times (recursion and multiple decodes)
    """
    losses = {}

    _, feed_dict = self._get_target_switch_to_test(feed_dict, training_handle, test_handle)
    testing_feed_dict = self._set_degrade_options(feed_dict)
    self._inference(test_step, testing_feed_dict)

    # Get the VC encoding
    # Note: In hierarchical VC, this would be the encoding of the final layer
    vc_encoding = self._component.get_vc().get_encoding()

    if self._opts['visualise_vc']:
      self._decoder(test_step, 'vc', 'vc', vc_encoding, testing_feed_dict)

    # Decode the DG using VC
    if self._opts['visualise_dg_at_vc']:
      dg_reconstruction = self._component.get_dg().get_decoding()
      self._decoder(test_step, 'dg', 'vc', dg_reconstruction, testing_feed_dict)

    # Decode the PC using VC, via DG
    if self._component.get_pc():

      dg_encoding = self._component.get_dg().get_encoding()
      pc_reconstruction = self._component.get_pc().get_decoding()

      # Decode the PC using DG
      if self._opts['visualise_pc_at_dg']:
        pc_decoded_at_dg = self._decoder(test_step, 'pc', 'dg', pc_reconstruction, testing_feed_dict)

        # Decode the decoded PC using VC to get the completed image
        if self._opts['visualise_pc_at_vc']:
          self._decoder(test_step, 'pc', 'vc', pc_decoded_at_dg, testing_feed_dict)

      # Calculate losses
      # ---------------------------------------------------------------------

      # pattern completion loss of the PC itself
      original = dg_encoding   # before degradation -------->      * WARNING assumes degradation after DG *
      completed = pc_reconstruction   # completed vector
      losses['completion_loss'] = np.square(original - completed).mean()

    return losses

  def _complete_pattern_isolate(self, feed_dict, training_handle, test_handle):
    """
    Exposes component to an incomplete pattern and calculates the completion loss.
    Calculate losses for each stage and the whole thing.
    - Iterates train set ONCE,
    - Iterates test set multiple times (recursion and multiple decodes)
    """
    losses = {}

    # Get the input image
    target_input = self._session.run(self._inputs, feed_dict={
        self._placeholders['dataset_handle']: training_handle
    })

    # Get the encoding of input image
    target_encoding = self._component.get_vc().get_encoding()

    # Switch to test set
    feed_dict.update({
        self._placeholders['dataset_handle']: test_handle
    })

    # Get placeholder to optionally degrade encoding
    degrade_hidden = False
    if self._opts['degrade_step'] == 'hidden':
      degrade_hidden = True
    degrade_hidden_pl = self._component.get_dual().get('degrade_hidden').get_pl()

    # End-to-end inference for pattern completion
    # ---------------------------------------------------------------------
    testing_feed_dict = feed_dict.copy()
    testing_feed_dict.update({
        degrade_hidden_pl: degrade_hidden
    })

    testing_fetches = {'labels': self._labels}
    self._component.add_fetches(testing_fetches, 'encoding')
    testing_fetched = self._session.run(testing_fetches, feed_dict=testing_feed_dict)
    self._component.set_fetches(testing_fetched, 'encoding')
    completed_encoding = self._component.get_pc().get_decoding()

    # Decode the completed encoding using VC
    # ---------------------------------------------------------------------
    vc_dual = self._component.get_dual('vc')
    vc_decode_pl = vc_dual.get('decode_pl').get_pl()
    vc_decode_feed_dict = feed_dict.copy()
    vc_decode_feed_dict.update({
        vc_decode_pl: completed_encoding
    })

    vc_decode_fetches = {}
    self._component.get_vc().add_fetches(vc_decode_fetches, 'decode')
    vc_decode_fetched = self._session.run(vc_decode_fetches, feed_dict=vc_decode_feed_dict)
    self._component.get_vc().set_fetches(vc_decode_fetched, 'decode')
    completed_input = vc_dual.get_values('decode_op')

    # Run the non-degraded VC encoding through PC
    # ---------------------------------------------------------------------
    if self._opts['degrade_step'] != 'none':
      pc_dual = self._component.get_dual('pc')

      # Encode the VC encoding using PC
      pc_encode_pl = pc_dual.get('encode_pl').get_pl()
      pc_encode_feed_dict = feed_dict.copy()
      pc_encode_feed_dict.update({
          pc_encode_pl: target_encoding
      })

      pc_encode_fetches = {}
      self._component.get_pc().add_fetches(pc_encode_fetches, 'encode')
      pc_encode_fetched = self._session.run(pc_encode_fetches, feed_dict=pc_encode_feed_dict)
      self._component.get_pc().set_fetches(pc_encode_fetched, 'encode')
      encoded_target_encoding = pc_dual.get_values('encode_op')

      # Decode the encoded VC encoding using PC
      pc_decode_pl = pc_dual.get('decode_pl').get_pl()
      pc_decode_feed_dict = feed_dict.copy()
      pc_decode_feed_dict.update({
          pc_decode_pl: encoded_target_encoding
      })

      pc_decode_fetches = {}
      self._component.get_pc().add_fetches(pc_decode_fetches, 'decode')
      pc_decode_fetches['pc'].pop('summaries')
      pc_decode_fetched = self._session.run(pc_decode_fetches, feed_dict=pc_decode_feed_dict)
      pc_decode_fetched['pc']['summaries'] = None
      self._component.get_pc().set_fetches(pc_decode_fetched, 'decode')
      reconstructed_encoding = pc_dual.get_values('decode_op')

      # Reconstruct the encoding back to the input
      if self._opts['degrade_step'] == 'input':
        vc_decode_feed_dict = feed_dict.copy()
        vc_decode_feed_dict.update({
            vc_decode_pl: reconstructed_encoding
        })

        vc_decode_fetches = {}
        self._component.get_vc().add_fetches(vc_decode_fetches, 'decode')
        vc_decode_fetches['vc'].pop('summaries')
        vc_decode_fetched = self._session.run(vc_decode_fetches, feed_dict=vc_decode_feed_dict)
        vc_decode_fetched['vc']['summaries'] = None
        self._component.get_vc().set_fetches(vc_decode_fetched, 'decode')
        reconstructed_input = vc_dual.get_values('decode_op')

    # Compute completion, reconstruction and total losses
    # ---------------------------------------------------------------------

    # Calculate completion loss
    if self._opts['degrade_step'] == 'input':
      losses['completion_loss'] = np.square(target_input - completed_input).mean()
    elif self._opts['degrade_step'] == 'hidden':
      losses['completion_loss'] = np.square(target_encoding - completed_encoding).mean()

    # Calculate the reconstruction loss
    if self._opts['degrade_step'] == 'input':
      losses['reconstruction_loss'] = np.square(target_input - reconstructed_input).mean()
    elif self._opts['degrade_step'] == 'hidden':
      losses['reconstruction_loss'] = np.square(target_encoding - reconstructed_encoding).mean()

    # Calculate total loss
    losses['total_loss'] = np.square(target_input - completed_input).mean()

    return losses
