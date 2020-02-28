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

"""EpisodicFewShotWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pagi.components.summarize_levels import SummarizeLevels
from pagi.utils import logger_utils, tf_utils, image_utils, data_utils
from pagi.utils.np_utils import print_simple_stats
from pagi.datasets.omniglot_dataset import OmniglotDataset

from aha.components.episodic_component import EpisodicComponent

from aha.datasets.omniglot_lake_dataset import OmniglotLakeDataset
from aha.datasets.omniglot_lake_runs_dataset import OmniglotLakeRunsDataset
from aha.datasets.omniglot_unseen_runs_dataset import OmniglotUnseenRunsDataset
from aha.datasets.omniglot_unseen_oneshot_dataset import OmniglotUnseenOneShotDataset

from aha.workflows.episodic_workflow import EpisodicWorkflow
from aha.workflows.pattern_completion_workflow import PatternCompletionWorkflow, UseTrainForTest

from aha.utils.generic_utils import compute_overlap, overlap_sample_batch
from aha.utils.few_shot_utils import compute_matching, create_and_add_comparison_image, add_completion_summary, \
  add_completion_summary_paper, mod_hausdorff_distance


match_mse_key = 'match_mse'
match_acc_mse_key = 'acc_mse'
sum_ambiguous_mse_key = 'amb_mse'

match_mse_tf_key = 'match_mse_tf'
match_acc_mse_tf_key = 'acc_mse_tf'
sum_ambiguous_mse_tf_key = 'amb_mse_tf'

match_olap_key = 'matching_matrices_olp'
match_olap_tf_key = 'matching_matrices_olp_tf'
match_acc_olap_key = 'acc_olp'
match_acc_olap_tf_key = 'acc_olp_tf'
sum_ambiguous_olap_key = 'amb_olp'
sum_ambiguous_olap_tf_key = 'amb_olp_tf'


class EpisodicFewShotWorkflow(EpisodicWorkflow, PatternCompletionWorkflow):
  """
  Few Shot Learning with the Episodic Component.

  modes:
  oneshot = classic lake test
  instance = like lake test, but identifying the same instance out of distractor exemplars from same class
  test_vc = visualise histograms of overlap
  create_data_subset = filter batch to create an 'idealised' batch where intra overlap is > inter overlap
  """

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        num_repeats=1,
        num_replays=1,
        consolidation_steps=0,
        random_recall=False,
        replay_buffer_threshold=0.0,
        superclass=False,
        class_proportion=1.0,
        invert_images=False,
        resize_images_factor=1.0,
        min_val=0,  # set any 0 in the input image, to this new min_val. ---> if >0, then don't do anything
        train_classes=['5', '6', '7', '8', '9'],
        test_classes=['5', '6', '7', '8', '9'],
        batch_all_classes=False,
        batch_no_duplicates=False,
        degrade_type='vertical',  # vertical, horizontal or random: the model completes image degraded by this method
        degrade_step='hidden',    # 'test', 'input', 'hidden' or 'none'
        degrade_factor=0.5,       # fraction to be degraded, if supported by the degrade_type option
        degrade_value=0,          # when degrading pixels, set them to this value
        noise_type='vertical',  # sp_float' or 'sp_binary'
        noise_step='test',  # 'test' or 'none'
        noise_factor=0.5,  # fraction of image to be noise
        noise_value=0,  # when degrading pixels, set them to this value
        completion_gain=1.0,
        train_recurse=False,
        test_recurse=False,
        recurse_iterations=0,
        rsummary_batches=2,
        input_mode={
            "train_first": "complete",
            "train_inference": "complete",
            "test_first": "complete",
            "test_inference": "complete"
        },
        evaluate=True,
        train=True,
        visualise_vc=False,          # show final vc decoded through vc (relevant for hierarchical VC)
        visualise_if_at_vc=False,    # show the IF encodings decoded through VC
        evaluate_mode=['oneshot', 'test_vc', 'test_dg'],
        evaluate_supermode='none',   # 'none' or 'same_train_and_test'    used for debugging, isolating units
        summarize_completion='none'  # the big matplotlib fig: 'none', 'to_file' or 'to_screen'

        """
        Notes on evaluate_mode:
        A list, containing non exclusive modes that can be: 
        'oneshot', 'instance', 'test_vc', 'test_dg', 'test_if' or 'create_data_subset'
        """
    )

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):
    super().__init__(session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
                     opts, summarize, seed, summary_dir, checkpoint_opts)
    self._training_features = {}
    self._testing_features = {}
    self._total_losses = {}               # dic for accumulating total (and then average) loss over all batches

    self._test_inputs = None
    self._summary_images = {}
    self._test_recurse_opts = self._opts['test_recurse']

    # dictionaries for all the overlap metrics (calculated by compute_overlap()
    self._vc_overlap = {}
    self._vc_core_overlap = {}
    self._dg_overlap = {}
    self._dg_overlap_cumulative = []

    # set these options
    self._is_overlap_per_label = False    # visualisation of overlap metrics per label or not

    self._add_comparison_images = False   # add the comparison images to the completion_summary if it is used
    self._paper = False                    # figures rendered for the paper

    self._replay_inputs = []
    self._replay_labels = []
    self._replay_step = 0
    self._consolidation_step = 0

    self._replay_inh = None
    self._big_loop = False

    self._all_replay_inputs = []
    self._all_replay_patterns = []
    self._all_replay_labels = []

    self._load_next_checkpoint = 0
    self._run_ended = True

  def _test_consistency(self):
    super()._test_consistency()

    if self.test_if_mode():
      if not self._hparams.use_interest_filter:
        raise RuntimeError("Using `test_if` mode, but not using InterestFilter!")

  def test_vc_mode(self):
    return 'test_vc' in self._opts['evaluate_mode']

  def test_if_mode(self):
    return 'test_if' in self._opts['evaluate_mode']

  def test_dg_mode(self):
    return 'test_dg' in self._opts['evaluate_mode']

  def instance_mode(self):
    return 'instance' in self._opts['evaluate_mode']

  def oneshot_mode(self):
    return 'oneshot' in self._opts['evaluate_mode']

  def create_data_subset_mode(self):
    return 'create_data_subset' in self._opts['evaluate_mode']

  def _is_eval_batch(self, batch):
    """
    If using PC with recall path, then we need to do many iterations of recall learning, before an 'evaluation'
    So, check if PC is ready for evaluation, otherwise, consider it an 'evaluation' batch.
    """
    if self._component.get_pc() is None:
      return True

    if not self._component.get_pc().use_nn_in_pr_path():
      return True

    # PC with recall path --> must be on num_repeats
    if (batch + 1) % self._opts['num_repeats'] == 0:
      return True

    return False

  def _is_decoding_vc_at_vc(self):
    return self._is_using_visualise_vc()

  def _is_decoding_pc_at_dg(self):
    if self._hparams.pc_type != 'none':
      if self._hparams.dg_type == 'fc':
        return True
    return False

  def _is_decoding_pc_at_vc(self):
    if self._hparams.pc_type != 'none':
      if self._hparams.dg_type != 'stub':
        if not self._hparams.use_interest_filter:
          return True
    return False

  def _is_decoding_if_at_vc(self):
    return self._is_using_if_at_vc()

  def _generate_datasets(self):
    """Override in child classes with your options"""
    same_train_and_test = UseTrainForTest.NO

    # some modes need same train and test
    if self.instance_mode():
      same_train_and_test = UseTrainForTest.IDENTICAL  # UseTrainForTest.SHUFFLED

    # option to override regardless
    if self._opts['evaluate_supermode'] == 'same_train_and_test':
      same_train_and_test = UseTrainForTest.IDENTICAL  # UseTrainForTest.SHUFFLED

    degrade_test = (self._opts['degrade_step'] == 'test')
    noise_test = (self._opts['noise_step'] == 'test')

    # Temporary fix for mismatched train/test inputs & labels
    # TODO(@abdel): Investigate this; as PAGI Decoder shouldn't be triggering the iterator
    additional_decode = 0
    if self._is_decoding_pc_at_dg():
      additional_decode += 1

    test_recurse_iterations = self._opts['recurse_iterations']

    if self._replay_mode():
      test_recurse_iterations += self._opts['num_replays'] * self._opts['recurse_iterations']

    train_dataset, test_dataset = self._gen_datasets_with_options(self._opts['train_classes'],
                                                                  self._opts['test_classes'],
                                                                  is_superclass=self._opts['superclass'],
                                                                  class_proportion=self._opts['class_proportion'],
                                                                  degrade_test=degrade_test,
                                                                  degrade_type=self._opts['degrade_type'],        # only relevant if degrade_test = True
                                                                  degrade_val=self._opts['degrade_value'],        # only relevant if degrade_test = True
                                                                  degrade_factor=self._opts['degrade_factor'],
                                                                  noise_test=noise_test,
                                                                  noise_type=self._opts['noise_type'],
                                                                  noise_factor=self._opts['noise_factor'],
                                                                  recurse_train=self._is_train_recursive(),
                                                                  recurse_test=self._is_inference_recursive(),
                                                                  num_batch_repeats=self._opts['num_repeats'],
                                                                  # recurse_iterations=self._opts['recurse_iterations'],
                                                                  recurse_iterations=test_recurse_iterations,
                                                                  additional_test_decodes=additional_decode,
                                                                  evaluate_step=self._opts['evaluate'],
                                                                  use_trainset_for_tests=same_train_and_test,
                                                                  invert_images=self._opts['invert_images'],
                                                                  min_val=self._opts['min_val'])
    return train_dataset, test_dataset

  def _create_dataset(self):
    if self._is_omniglot_lake():
      if self._dataset_type.__name__ == OmniglotLakeDataset.__name__:
        self._dataset = self._dataset_type(self._dataset_location,
                                           self._hparams.batch_size,
                                           self._opts['test_classes'],
                                           self.instance_mode())
      elif self._dataset_type.__name__ == OmniglotLakeRunsDataset.__name__:
        self._dataset = self._dataset_type(self._dataset_location,
                                           self._hparams.batch_size)
      elif self._dataset_type.__name__ == OmniglotUnseenRunsDataset.__name__:
        self._dataset = self._dataset_type(self._dataset_location,
                                           self._hparams.batch_size)
      elif self._dataset_type.__name__ == OmniglotUnseenOneShotDataset.__name__:
        self._dataset = self._dataset_type(self._dataset_location,
                                           self._hparams.batch_size)
      else:
        self._dataset = self._dataset_type(self._dataset_location)
    else:
      self._dataset = self._dataset_type(self._dataset_location)

  def _setup_recursive_train_modes(self, batch_type):
    """ Set the appropriate mode depending on batch type.
    This is only called when recursive training """

    mode = 'training'
    # If each component is in encoding mode
    if all(v == 'encoding' for v in batch_type.values()):
      mode = 'inference'
    return mode

  def _setup_checkpoint_saver(self):
    """Handles the saving and restoration of graph state and variables."""

    if not self._replay_mode():
      return super()._setup_checkpoint_saver()

    max_to_keep = self._export_opts['max_to_keep']

    # First call to this function by setup()
    if self._load_next_checkpoint == 0:
      self._saver = tf.train.Saver(max_to_keep=max_to_keep)
      self._last_step = 0
      return

    filename = 'model.ckpt-' + str(self._load_next_checkpoint)
    checkpoint_path = os.path.join(self._checkpoint_opts['checkpoint_path'], filename)
    print('Checkpoint Path =', checkpoint_path)

    # Loads a subset of the checkpoint, specified by variable scopes
    if self._checkpoint_opts['checkpoint_load_scope']:
      load_scope = []
      init_scope = []

      scope_list = self._checkpoint_opts['checkpoint_load_scope'].split(',')
      for i, scope in enumerate(scope_list):
        scope_list[i] = scope.lstrip().rstrip()

      global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      for variable in global_variables:
        # Add any variables that match the specified scope to a separate list
        # Note: global_step is excluded and re-initialised, even if within scope
        if variable.name.startswith(tuple(scope_list)) and 'global_step' not in variable.name:
          load_scope.append(variable)
        else:
          init_scope.append(variable)

      # Load variables from specified scope
      saver = tf.train.Saver(load_scope)
      self._restore_checkpoint(saver, checkpoint_path)

      # Re-initialise any variables outside specified scope, including global_step
      init_scope_op = tf.variables_initializer(init_scope, name='init')
      self._session.run(init_scope_op)

      # Switch to encoding mode if freezing loaded scope
      if self._checkpoint_opts['checkpoint_frozen_scope']:
        self._freeze_training = True

      # Create new tf.Saver to save new checkpoints
      self._saver = tf.train.Saver(max_to_keep=max_to_keep)
      self._last_step = 0

    # Otherwise, attempt to load the entire checkpoint
    else:
      self._saver = tf.train.Saver(max_to_keep=max_to_keep)
      self._last_step = self._restore_checkpoint(self._saver, checkpoint_path)

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""

    # Training
    # -------------------------------------------------------------------------
    training_handle = self._session.run(self._dataset_iterators['training'].string_handle())
    self._session.run(self._dataset_iterators['training'].initializer)

    if evaluate:
      test_handle = self._session.run(self._dataset_iterators['test'].string_handle())
      self._session.run(self._dataset_iterators['test'].initializer)

    self._on_before_training_batches()

    # set some hyperparams to instance variables for access in train and complete methods
    # (to be compatible with base class method signatures)
    self._rsummary_from_batch = num_batches - self._opts['rsummary_batches']  # recursive sums for last n batches

    self._input_mode = self._opts['input_mode']

    for batch in range(num_batches):
      if self._replay_mode() and self._run_ended:
        self._load_next_checkpoint += 40
        self._setup_checkpoint_saver()
        self._run_ended = False

      logging.debug("----------------- Batch: %s", str(batch))
      feed_dict = {}

      if train:
        global_step = tf.train.get_global_step(self._session.graph)
        if global_step is not None:
          training_step = self._session.run(global_step)
          training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)
        else:
          training_step = 0
          training_epoch = 0

        # Perform the training, and retrieve feed_dict for evaluation phase
        logging.debug("\t----------------- Train with training_step: %s", str(training_step))
        feed_dict, _ = self.training(training_handle, batch)

        self._on_after_training_batch(batch, training_step, training_epoch)

        # Export any experiment-related data
        # -------------------------------------------------------------------------
        if self._export_opts['export_filters']:
          if (batch == num_batches - 1) or ((batch + 1) % self._export_opts['interval_batches'] == 0):
            self.export(self._session, feed_dict)

        if self._export_opts['export_checkpoint']:
          if (batch == num_batches - 1) or ((batch + 1) % self._export_opts['interval_batches'] == 0):
            self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch + 1)

      if evaluate:
        logging.debug("----------------- Complete with training_step: %s", str(batch))
        losses = self._complete_pattern(feed_dict, training_handle, test_handle, batch)

        self._on_after_evaluate(losses, batch)

      finished_recall = False
      if self._build_replay_dataset():
        if (self._replay_step + 1) % self._opts['num_replays'] == 0:
          print('Finish Recall, starting consolidation...\n')
          self._consolidation_step = 0
          finished_recall = True

        self._replay_step += 1

      if not finished_recall and self._consolidation_mode():
        if (self._consolidation_step + 1) % self._opts['consolidation_steps'] == 0:
          print('Finished consolidation, starting next run...\n')

          self._replay_step = 0

          self._big_loop = False
          self._run_ended = True

          self._replay_inputs = []
          self._replay_labels = []
          self._all_replay_inputs = []
          self._all_replay_labels = []

        self._consolidation_step += 1

    # this is here because we want it to execute at the very end of all batches (training or evaluation)
    if len(self._total_losses.keys()) > 0:
      print("\n--------- Averages for all batches: ----------")
      for accuracy_type, vals in self._total_losses.items():
        av = np.mean(vals, dtype=np.float64)
        print("\t{}: {}     (length={})".format(accuracy_type, av, len(vals)))
        logger_utils.log_metric("av__{}".format(accuracy_type), av)

      # print as comma separated list for import into a spreadsheet
      # headings
      for accuracy_type, vals in self._total_losses.items():
        print("{}, ".format(accuracy_type), end='')
      print('\n')
      # values
      for accuracy_type, vals in self._total_losses.items():
        av = np.mean(vals, dtype=np.float64)
        print("{}, ".format(av), end='')
      print('\n')

  def _replay_mode(self):
    return 'replay' in self._opts['evaluate_mode']

  def _replay_preprocess(self, inputs, labels):
    if not isinstance(inputs, np.ndarray):
      inputs = np.array(inputs)
      labels = np.array(labels)
      inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
    return inputs, labels

  def _build_replay_dataset(self):
    return self._replay_mode() and self._replay_step in list(range(self._opts['num_replays']))

  def _consolidation_mode(self):
    return self._replay_mode() and not self._build_replay_dataset()

  def _random_sample(self, inputs, labels):
    """Random sample with duplicates."""
    batch_inputs = []
    batch_labels = []

    size = len(inputs)

    for i in range(self._hparams.batch_size):
      idx = np.random.choice(np.arange(size), 1, replace=False)[0]
      batch_inputs.append(inputs[idx])
      batch_labels.append(labels[idx])

    return np.array(batch_inputs), np.array(batch_labels)

  def _setup_train_feed_dict(self, batch_type, training_handle):
    feed_dict = super()._setup_train_feed_dict(batch_type, training_handle)

    if self._consolidation_mode():
      print('\n')
      print('Consolidation Step -', self._consolidation_step)

      if self._all_replay_inputs:
        assert len(self._all_replay_inputs) == len(self._all_replay_labels)

        random_sample = True

        if random_sample:
          replay_inputs, replay_labels = self._replay_preprocess(self._all_replay_inputs, self._all_replay_labels)

          batch_inputs, batch_labels = self._random_sample(replay_inputs, replay_labels)
        else:
          batch_inputs = np.array(self._all_replay_inputs)
          batch_labels = np.array(self._all_replay_labels)

        feed_dict.update({
            self._component.get_dual().get_pl('replay'): True,

            self._component.get_dual().get_pl('replay_inputs'): batch_inputs,
            self._component.get_dual().get_pl('replay_labels'): batch_labels
        })
      else:
        print('Using groundtruth data for consolidation...')

    return feed_dict

  def training(self, training_handle, training_step, training_fetches=None):
    """The training procedure within the batch loop"""

    # If we're in replay mode, and this is the first step of the run, then build the replay dataset first
    if self._build_replay_dataset():
      print('\n')
      print('Replay Step =', self._replay_step)
      print('Number of samples =', len(self._all_replay_inputs))

      fetched = {}
      batch_type = self._setup_train_batch_types()
      feed_dict = self._setup_train_feed_dict(batch_type, training_handle)

      # Trigger the iterator since we're skipping a training step
      training_fetches = ({'labels': self._labels})
      self._session.run(training_fetches, feed_dict)

      return feed_dict, fetched

    # re-initialise variables if we're starting a new run
    if self._is_eval_batch(training_step - 1) and not self._replay_mode():  # previous step was an 'evaluation' step
      self._reinitialize_networks()

    self._training_features = {}

    if training_fetches is None:
      training_fetches = {}

    training_fetches.update({'labels': self._labels})
    feed_dict, fetched = super().training(training_handle, training_step, training_fetches)

    self._training_features = self._extract_features(fetched)

    if self._component.is_build_pc():
      self._training_features['pc'] = self._component.get_pc().get_input('training')      # target output is value to be memorised (the training input)
      # self._training_features['pc_in'] = self._component.get_pc().get_input('encoding')   # NOTE: this is the output of the PR on the training set
      self._training_features['pc_in'] = self._component.get_pc().get_input('training')   # NOTE: this is the target (provided by DG), this is not the PR output because it is in 'training' mode.

    logging.debug("**********>> Training: Batch={},  Training labels = {}".format(training_step, self._training_features['labels'][0]))

    input_images = self._component.get_vc().get_inputs()
    self._prep_for_summaries_after_training(input_images)

    return feed_dict, fetched

  def _complete_pattern(self, feed_dict, training_handle, test_handle, test_step):
    """Shows component new examples to find 'matches' with the train batch."""

    # 1) Preparations
    # --------------------------------------------------------------
    losses = {}

    target_inputs, testing_feed_dict = self._get_target_switch_to_test(feed_dict, training_handle, test_handle)
    testing_feed_dict = self._set_degrade_options(testing_feed_dict)

    # Don't bother recursions in Hopfield while we're learning the 'fc network' to map VC to PC
    # Note: we still need to do this 'evaluation' step

    if self._is_eval_batch(test_step) or self._build_replay_dataset():
      self._opts['test_recurse'] = self._test_recurse_opts
    else:
      self._opts['test_recurse'] = False

    if self._consolidation_mode():
      testing_feed_dict.update({
          self._component.get_dual().get_pl('replay'): False,
          self._component.get_pc().get_dual().get_pl('random_recall'): False
      })

    # 2) Run AMTL on test set (when enabled, will get PC completion of input image)
    # --------------------------------------------------------------
    testing_fetches = {}
    testing_fetches.update(
        {
            'labels': self._labels,
            'test_inputs': self._inputs
        }
    )

    threshold = self._opts['replay_buffer_threshold']
    use_threshold = threshold > 0
    random_recall = self._opts['random_recall']

    big_loop = self._big_loop
    big_loop_done = False

    if self._build_replay_dataset():
      vc_encoding, _ = self._component.get_signal('vc')
      vc_encoding_shape = vc_encoding.get_shape().as_list()
      random_noise_shape = self._component.get_pc().get_dual().get_pl('random_noise').get_shape().as_list()

      random_noise = np.random.uniform(-1, 1, random_noise_shape)
      # print('Random Noise =', random_noise_shape, np.min(random_noise), np.max(random_noise))

      if big_loop and self._replay_inputs:
        print('Replay Mode =', 'Big-Loop Recurrence')

        self._replay_inputs = np.array(self._replay_inputs)
        self._replay_labels = np.array(self._replay_labels)

        batch_inputs, batch_labels = self._replay_preprocess(self._replay_inputs, self._replay_labels)

        testing_feed_dict.update({
            self._component.get_dual().get_pl('replay'): True,
            self._component.get_pc().get_dual().get_pl('random_recall'): False,
            self._component.get_pc().get_dual().get_pl('use_inhibition'): False,
            self._component.get_pc().get_dual().get_pl('random_noise'): random_noise,
            self._component.get_dual().get_pl('replay_inputs'): batch_inputs,
            self._component.get_dual().get_pl('replay_labels'): batch_labels
        })

        big_loop_done = True
        self._big_loop = False

      elif random_recall:
        print('Replay Mode =', 'Random Recall')

        testing_feed_dict.update({
            self._component.get_pc().get_dual().get_pl('random_recall'): True,
            self._component.get_pc().get_dual().get_pl('use_inhibition'): False,
            self._component.get_pc().get_dual().get_pl('random_noise'): random_noise,
            self._component.get_pc().get_dual().get_pl('inhibition'): np.zeros(vc_encoding_shape)
        })

        self._big_loop = True

      self._replay_inputs = []
      self._replay_labels = []

    logging.debug("**********>> Testing: Batch={}".format(test_step))
    testing_fetched = self._inference(test_step, testing_feed_dict, testing_fetches)

    self._testing_features = self._extract_features(testing_fetched)
    self._test_inputs = testing_fetched['test_inputs']

    # from skimage.measure import compare_ssim

    if self._build_replay_dataset():
      replay_inputs = np.reshape(testing_fetched['pc']['ec_out_raw'], self._dataset.shape)
      replay_images = replay_inputs

      replay_labels = testing_fetched['ll_pc']['preds']
      replay_patterns = testing_fetched['pc']['y']

      for i, (image, label) in enumerate(zip(replay_inputs, replay_labels)):
        append = False

        if use_threshold and np.sum(image) > threshold:
          # If not in big loop, always append
          if not big_loop:
            append = False

          # If in big loop, only append when big loop is done
          elif big_loop and big_loop_done:
            # append = False
            append = True

            # # Only append labels that match groundtruth
            # if np.argmax(replay_labels[i]) in testing_fetched['labels']:
            #   append = True

            # # n = 5
            # # top_n = replay_labels[i].argsort()[-n:][::-1]
            # # print(i, top_n, replay_labels[i][top_n])

            # # Avoid duplicate images
            # if append and self._all_replay_labels:
            #   append = False

            #   # for bfr_img, bfr_label in zip(self._all_replay_inputs, self._all_replay_labels):
            #   #   print(bfr_image.shape, np.squeeze(bfr_img).shape)

            #   #   (score, diff) = compare_ssim(np.squeeze(bfr_img), np.squeeze(image), full=True)
            #   #   print('BFR Label =', bfr_label, 'Current label =', label)
            #   #   print('SSIM =', score, diff)

            #   # The second condition makes it easier to track; but requires knowledge of training labels
            #   if np.argmax(replay_labels[i]) not in np.argmax(np.array(self._all_replay_labels), axis=1)[:]:
            #     append = True

            # Limit buffer to batch_size length
            # Note: Maybe we don't need to limit this.
            # if len(self._replay_labels) >= self._hparams.batch_size:
            #   append = False

        elif not use_threshold:
          append = True

        if append:
          self._all_replay_inputs.append(replay_inputs[i])
          self._all_replay_patterns.append(replay_patterns[i])
          self._all_replay_labels.append(replay_labels[i])

          # print('Sample # =', i, 'Label = ', np.max(label), np.argmax(label))

        self._replay_inputs.append(replay_inputs[i])
        self._replay_labels.append(replay_labels[i])

        # # Do a big-loop iteration after we collect all images or on final replay step
        # if (self._replay_step + 1) % (self._opts['num_replays'] - 1) == 0 or (
        #     len(self._all_replay_inputs) == self._hparams.batch_size):
        #   self._big_loop = True

      # interval = self._opts['num_replays']
      interval = 1

      if self._all_replay_labels:
        labels_retrieved = np.argmax(self._all_replay_labels, axis=1)
        labels_missing = np.setdiff1d(testing_fetched['labels'], labels_retrieved)
        labels_unique = np.unique(labels_retrieved)

      pc_in_train = self._component.get_pc().get_input('training')
      pc_in_test = self._component.get_pc().get_input('encoding')
      _, pc_input_shape = self._component.get_signal('pc_input')
      print(pc_input_shape)

      if (self._replay_step + 1) % interval == 0 and self._all_replay_labels:

        print('Labels retrieved ({0}) = {1}'.format(len(labels_retrieved), labels_retrieved))
        print('Labels missing ({0}) = {1}'.format(len(labels_missing), labels_missing))
        print('Groundtruth =', testing_fetched['labels'])
        print('Unique =', labels_unique)

        plt.switch_backend('agg')

        rows = 7
        cols = self._hparams.batch_size
        _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=[10, 2], num=test_step)
        plt.subplots_adjust(left=0, right=1.0, bottom=0, top=1.0, hspace=0.1, wspace=0.1)

        # plot simple raster image on each sub-plot
        for i, ax in enumerate(ax.flat):
          # get indices of row/column
          row_idx = i // cols
          col_idx = i % cols

          img_idx = col_idx

          if row_idx in [1, 3]:
            label_idx = testing_fetched['labels'][img_idx]
            label_real = self._dataset.eval_classes[label_idx]
            ax.text(0.3, 0.3, str(label_real))
          elif row_idx == 5:
            label_idx = np.argmax(replay_labels[img_idx])
            label_real = self._dataset.eval_classes[label_idx]
            ax.text(0.3, 0.3, str(label_real))
          else:
            if row_idx == 0:
              img = target_inputs[img_idx]
            elif row_idx == 2:
              img = self._test_inputs[img_idx]
            elif row_idx == 4:
              img = replay_images[img_idx]
            elif row_idx == 6:
              img = replay_patterns[img_idx]

            if row_idx in [0, 2, 4]:
              image_shape = [replay_images.shape[1], replay_images.shape[2]]
              img = np.reshape(img, image_shape)
              ax.imshow(img, cmap='binary', vmin=0, vmax=1)
            else:
              img = np.reshape(img, [pc_input_shape[1], pc_input_shape[2]])
              ax.imshow(img, vmin=-1, vmax=1)

          ax.axis('off')

        filetype = 'png'
        filename = 'batch_replay_summary_' + str(test_step) + '.' + filetype
        filepath = os.path.join(self._summary_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.2, dpi=300, format=filetype)
        plt.clf()

        rows = 3
        cols = len(self._all_replay_inputs)
        _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=[10, 2], num=test_step)
        plt.subplots_adjust(left=0, right=1.0, bottom=0, top=1.0, hspace=0.1, wspace=0.1)

        # plot simple raster image on each sub-plot
        for i, ax in enumerate(ax.flat):
          # get indices of row/column
          row_idx = i // cols
          col_idx = i % cols

          img_idx = col_idx

          if row_idx == 1:
            label_idx = np.argmax(self._all_replay_labels[img_idx])
            label_real = self._dataset.eval_classes[label_idx]
            ax.text(0.3, 0.3, str(label_real), fontsize=9)
          elif row_idx == 0:
            img = self._all_replay_inputs[img_idx]

            image_shape = [replay_images.shape[1], replay_images.shape[2]]
            img = np.reshape(img, image_shape)
            ax.imshow(img, cmap='binary', vmin=0, vmax=1)
          elif row_idx == 2:
            img = self._all_replay_patterns[img_idx]

            img = np.reshape(img, [pc_input_shape[1], pc_input_shape[2]])
            ax.imshow(img, vmin=-1, vmax=1)

          ax.axis('off')

        filetype = 'png'
        filename = 'filtered_replay_summary_' + str(test_step) + '.' + filetype
        filepath = os.path.join(self._summary_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.2, dpi=300, format=filetype)

    logging.debug("          --------> Batch={},  Testing labels = {}".format(test_step, self._testing_features['labels'][0]))

    if not self._is_eval_batch(test_step):
      return losses

    summarise = True
    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      summarise = False

    # 3) Visualise VC encodings in Image space (useful for hierarchical VC)
    # --------------------------------------------------------------
    vc_encoding_test = self._testing_features['vc']
    test_labels = self._testing_features['labels']

    if self._opts['visualise_vc']:
      self._decoder(test_step, 'vc', 'vc', vc_encoding_test, testing_feed_dict, summarise=summarise)

    # 4) Visualise Interest Filter encodings in Image space
    # --------------------------------------------------------------
    if self.test_if_mode() and self._is_using_if_at_vc():
      me = self._component.get_interest_filter_masked_encodings()
      pe = self._component.get_interest_filter_positional_encodings()
      self._decoder(test_step, 'if_masked_encodings', 'vc', me, testing_feed_dict, summarise=summarise)
      self._decoder(test_step, 'if_positional_encodings', 'vc', pe, testing_feed_dict, summarise=summarise)

    # 5) Analyse VC
    # --------------------------------------------------------------
    if self.test_vc_mode():
      vc_encoding_train = self._training_features['vc']
      train_labels = self._training_features['labels']

      self._vc_overlap = {}
      compute_overlap(self._vc_overlap, vc_encoding_train, train_labels, vc_encoding_test, test_labels,
                      per_label=self._is_overlap_per_label)

      losses['vc_accuracy'] = self._vc_overlap['accuracy']
      losses['vc_confusion'] = self._vc_overlap['confusion']
      self._report_average_metric('test_vc_accuracy', self._vc_overlap['accuracy'])

      # if testing Interest Filter as well, we also want to see the vc component overlap before interest filtering
      if self.test_if_mode():
        vc_core_encoding_train = self._training_features['vc_core_hidden']
        vc_core_encoding_test = self._testing_features['vc_core_hidden']
        train_labels = self._training_features['labels']

        self._vc_core_overlap = {}
        compute_overlap(self._vc_core_overlap, vc_core_encoding_train, train_labels, vc_core_encoding_test, test_labels,
                        per_label=self._is_overlap_per_label)

        losses['vc_core_accuracy'] = self._vc_core_overlap['accuracy']
        losses['vc_core_confusion'] = self._vc_core_overlap['confusion']
        self._report_average_metric('test_vc_core_accuracy', self._vc_core_overlap['accuracy'])

    # 6) Analyse DG
    # --------------------------------------------------------------
    if self.test_dg_mode():
      dg_encoding_train = self._training_features['dg_hidden']
      train_labels = self._training_features['labels']

      self._dg_overlap = {}
      # dg_as_binary = np.greater(dg_encoding_train, 0.0).astype(float)

      compute_overlap(self._dg_overlap,
                      dg_encoding_train, train_labels,
                      None, None,
                      per_label=False)

      self._dg_overlap_cumulative.append(self._dg_overlap['inter'])

    # 7) Analyse whole AMTL for instance or oneshot mode
    # --------------------------------------------------------------
    # express PC completion decoded through DG then VC (if possible)
    # if sub components present, get values, otherwise set to none
    # this makes it flexible, it will work and populate whatever it can

    if self.instance_mode() or self.oneshot_mode():
      if not self._component.is_build_dg() and not self._component.is_build_pc():
        raise RuntimeError("You need both PC and DG for running `oneshot` or `fewshot` as intended.")

    if 'simple' not in self._opts['evaluate_mode']:
      pc_completion = None
      pc_at_dg = None
      pc_at_vc = None
      pc_in = None

      if self._component.get_pc() is not None:
        pc_completion = self._component.get_pc().get_decoding()                              # equiv to dg hidden
        pc_in = self._component.get_pc().get_input("encoding")          # the output of cue_nn, input to PC itself

        # If not using DG, feed PC decoding directly to VC
        pc_at_vc_input = pc_completion

        if self._is_decoding_pc_at_dg():
          pc_at_dg = self._decoder(test_step, 'pc', 'dg', pc_completion, testing_feed_dict,
                                   summarise=summarise)  # equiv to dg recon/vc hidden
          pc_at_vc_input = pc_at_dg  # Feed DG decoding to VC, instead of PC

        if self._is_decoding_pc_at_vc():
          # this doesn't make much sense if we are using the interest filter (and dimensions don't match)
          if not self._hparams.use_interest_filter:
            pc_at_vc = self._decoder(test_step, 'pc', 'vc', pc_at_vc_input, testing_feed_dict,
                                     summarise=summarise)     # equiv to vc recon

      # 8) Analyse classification performance with and without AMTL
      # --------------------------------------------------------------
      if self._component.is_build_ll_vc():
        losses['ll_vc_accuracy'] = self._component.get_ll_vc().get_values('accuracy')
        losses['ll_vc_accuracy_unseen'] = self._component.get_ll_vc().get_values('accuracy_unseen')

        self._report_average_metric('ll_vc_accuracy', losses['ll_vc_accuracy'])
        self._report_average_metric('ll_vc_accuracy_unseen', losses['ll_vc_accuracy_unseen'])

      if self._component.is_build_ll_pc():
        losses['ll_pc_accuracy'] = self._component.get_ll_pc().get_values('accuracy')
        losses['ll_pc_accuracy_unseen'] = self._component.get_ll_pc().get_values('accuracy_unseen')

        self._report_average_metric('ll_pc_accuracy', losses['ll_pc_accuracy'])
        self._report_average_metric('ll_pc_accuracy_unseen', losses['ll_pc_accuracy_unseen'])

      if self._component.is_build_ll_ensemble():
        losses['ll_ensemble_accuracy'] = self._component.get_values('ensemble_accuracy')
        losses['ll_ensemble_accuracy_unseen'] = self._component.get_values('ensemble_accuracy_unseen')

        self._report_average_metric('ll_ensemble_accuracy', losses['ll_ensemble_accuracy'])
        self._report_average_metric('ll_ensemble_accuracy_unseen', losses['ll_ensemble_accuracy_unseen'])

      if self._component.get_pc().use_pm_raw is True:
        vc_input = target_inputs
        ec_out_raw = self._component.get_pc().get_ec_out_raw()

        vc_input_flat = np.reshape(vc_input, [vc_input.shape[0], np.prod(vc_input.shape[1:])])
        ec_out_raw_flat = np.reshape(ec_out_raw, [ec_out_raw.shape[0], np.prod(ec_out_raw.shape[1:])])

        # print('vc stats', np.min(vc_input_flat), np.max(vc_input_flat))
        # print('pm_raw stats (before)', np.min(ec_out_raw_flat), np.max(ec_out_raw_flat))
        ec_out_raw_flat = np.clip(ec_out_raw_flat, a_min=np.min(vc_input_flat), a_max=np.max(vc_input_flat))
        # print('pm_raw stats (after)', np.min(ec_out_raw_flat), np.max(ec_out_raw_flat))

        pm_raw_mse = np.square(vc_input_flat - ec_out_raw_flat).mean()

        pm_raw_mhd = mod_hausdorff_distance(vc_input_flat, ec_out_raw_flat)

        losses['acc_mse_pm_raw'] = pm_raw_mse
        losses['acc_mhd_pm_raw'] = pm_raw_mhd

        self._report_average_metric('acc_mse_pm_raw', pm_raw_mse)
        self._report_average_metric('acc_mhd_pm_raw', pm_raw_mhd)

      modes = self._opts['evaluate_mode']
      self._compute_few_shot_metrics(losses, modes, pc_in, pc_completion, pc_at_dg, pc_at_vc)
      self._prep_for_summaries_after_completion(self._test_inputs,
                                                with_comparison_images=self._add_comparison_images)    # prepare for more comprehensive summaries

      if pc_at_vc is not None:
        losses['loss_pc_at_vc'] = np.square(self._test_inputs - pc_at_vc).mean()

    if self._consolidation_mode() and  self._all_replay_inputs:
      labels_retrieved = np.argmax(self._all_replay_labels, axis=1)
      labels_missing = np.setdiff1d(testing_fetched['labels'], labels_retrieved)
      labels_unique = np.unique(labels_retrieved)

      losses['recalled_unseen'] = int(testing_fetched['labels'][0] in labels_retrieved)
      losses['recalled_batch'] = int(len(labels_missing) == 0)  # pylint: disable=len-as-condition
      losses['num_unique_recalled'] = len(labels_unique)
      losses['num_samples_recalled'] = len(labels_retrieved)
      losses['num_samples_missing'] = len(labels_missing)

      self._report_average_metric('recalled_unseen', losses['recalled_unseen'])
      self._report_average_metric('recalled_batch', losses['recalled_batch'])
      self._report_average_metric('num_unique_recalled', losses['num_unique_recalled'])
      self._report_average_metric('num_samples_recalled', losses['num_samples_recalled'])
      self._report_average_metric('num_samples_missing', losses['num_samples_missing'])

    return losses

  def _on_after_evaluate(self, results, batch):
    if 'simple' in self._opts['evaluate_mode']:
      return

    console = True
    matching_matrix_keys = [match_mse_key, match_mse_tf_key, match_olap_key, match_olap_tf_key]
    matching_accuracies_keys = [match_acc_mse_key, match_acc_mse_tf_key, match_acc_olap_key, match_acc_olap_tf_key]
    sum_ambiguous_keys = [sum_ambiguous_mse_key, sum_ambiguous_mse_tf_key, sum_ambiguous_olap_key, sum_ambiguous_olap_tf_key]

    if self.create_data_subset_mode():
      self._create_data_subset()

    # 0) logs that match 'after training' logs, to show training vs testing losses
    if self._component.get_pc():
      pc = self._component.get_pc()
      loss_pc = pc.get_loss()
      loss_pm = 0
      loss_pm_raw = 0
      loss_pr = 0

      # variable names
      output = '|PC Losses (memorise'
      if pc.use_input_cue:
        loss_pr = pc.get_loss_pr()
        output = output + ', PR'
      if pc.use_pm or pc.use_pm_raw:
        loss_pm, loss_pm_raw = pc.get_losses_pm()
        output = output + ', PM, PM_raw'

      # values
      output = output + ') = ('
      output = output + '{:.2f}'.format(loss_pc)
      if pc.use_input_cue:
        output = output + ', {:.2f}'.format(loss_pr)
      if pc.use_pm or pc.use_pm_raw:
        output = output + ', {:.2f}, {:.2f}'.format(loss_pm, loss_pm_raw)
      output = output + ')|'

      if (batch+1) % self._logging_freq == 0:
        logging.info(output)

    # print("Train Labels: " + str(self._training_features['labels']))
    # print("Test Labels: " + str(self._testing_features['labels']))

    if not self._is_eval_batch(batch):
      return

    # setup some structures for following sections
    skip_console = matching_matrix_keys + matching_accuracies_keys + sum_ambiguous_keys
    skip_logging = skip_console + ['vc_confusion', 'vc_core_confusion']
    skip_summaries = skip_console + skip_logging

    if self._is_omniglot_lake() and self._component.batch_size() > 15:
      skip_console.extend(['vc_confusion', 'vc_core_confusion'])        # too many classes to make this legible

    # 1) print metrics to console
    # ------------------------------------------------------------
    # console
    if console:

      print("Train Labels: " + str(self._training_features['labels']))
      print("Test Labels: " + str(self._testing_features['labels']))

      print("\n--------- General Metrics -----------")
      np.set_printoptions(threshold=np.inf)
      for metric, metric_value in results.items():
        if metric not in skip_console:
          print("\t{0} : {1}".format(metric, metric_value))

      print("\n--------- Oneshot/Lake Metrics (i.e. PC fed back through AMTL) -----------")
      for matching_accuracies_key in matching_accuracies_keys:    # for different comparison types
        matching_accuracies = results[matching_accuracies_key]
        for accuracy_type, val in matching_accuracies.items():    # for different features
          print("\t{}_{} : {:.3f}".format(matching_accuracies_key, accuracy_type, val))

      for sum_ambiguous_key in sum_ambiguous_keys:   # for different comparison types
        sum_ambiguous = results[sum_ambiguous_key]
        for sum_ambiguous_type, val in sum_ambiguous.items():   # for different feature types
          print("\t{}_{} : {:.3f}".format(sum_ambiguous_key, sum_ambiguous_type, val))

      if self.test_dg_mode():
        if 'inter' in self._dg_overlap:
          inter = self._dg_overlap['inter']
          print("\n------- Test DG Orthogonality -------")
          print("Max possible overlap = {}".format(self._hparams.dg_fc_sparsity))
          print_simple_stats(inter, 'DG-Inter', verbose=True)
          print_simple_stats(inter, 'DG-Inter normed', verbose=True, normalise_by=self._hparams.dg_fc_sparsity)
          #print('DG Olap cum. ', self._dg_overlap_cumulative)
          print_simple_stats(self._dg_overlap_cumulative, 'Cum. DG-Inter', verbose=True)

      if self.test_vc_mode():
        print("\n------- Test VC  -------")
        print_simple_stats(self._vc_overlap['accuracy_margin'], 'Accuracy Margin', verbose=True)
        print_simple_stats(self._vc_overlap['inter'], 'Inter', verbose=True)
        print_simple_stats(self._vc_overlap['intra'], 'Intra', verbose=True)

        if self.test_if_mode():
          print("\n------- Test VC Core  -------")
          print_simple_stats(self._vc_core_overlap['accuracy_margin'], 'Accuracy Margin', verbose=True)
          print_simple_stats(self._vc_core_overlap['inter'], 'Inter', verbose=True)

    # 2) logs for tabulating results (i.e. mlflow)
    # ------------------------------------------------------------
    # the few shot metrics
    for metric, metric_value in results.items():
      if metric not in skip_logging:
        logger_utils.log_metrics({metric: metric_value})

    # and other values for testing
    if self._component.get_pc() is not None and self._component.get_pc().use_input_cue:
      if loss_pr is not None:
        logger_utils.log_metric('loss_recall_test', loss_pr)
        logger_utils.log_metric('loss_recall_train', self._train_loss_pr)

      # if final value for a run, then we want to see what it is, and the average for all runs
      if self._is_eval_batch(batch):
        if loss_pr is not None:
          self._report_average_metric('loss_recall_test', loss_pr)
          self._report_average_metric('loss_recall_train', self._train_loss_pr)

      # the Oneshot/Lake results (i.e. after output from pc calc match at different levels of architecture)
      for matching_accuracies_key in matching_accuracies_keys:  # for different comparison types
        matching_accuracies = results[matching_accuracies_key]
        for accuracy_type, val in matching_accuracies.items():  # for different features
          logger_utils.log_metric("{}".format(matching_accuracies_key), val)

      for sum_ambiguous_key in sum_ambiguous_keys:              # for different comparison types
        sum_ambiguous = results[sum_ambiguous_key]
        for sum_ambiguous_type, val in sum_ambiguous.items():   # for different features
          logger_utils.log_metric("{}".format(sum_ambiguous_type), val)

    # 3) summaries
    # ------------------------------------------------------------
    if self._summarize:

      # ---- Test VC Summaries
      summary = tf.Summary()
      if self.test_vc_mode():
        summary = self._add_test_vc_summaries(self._vc_overlap, batch)
        self._writer.add_summary(summary, batch)

        # If also test Interest Filter, see vc core olap (the SCAE before Interest Filter and pooling, norming etc)
        if self.test_if_mode():
          summary = self._add_test_vc_summaries(self._vc_core_overlap, batch, fig_scope="core")
          self._writer.add_summary(summary, batch)

      # ---- Test DG Summaries
      if self.test_dg_mode():
        summary = self._add_test_dg_summaries(self._dg_overlap, batch)
        self._writer.add_summary(summary, batch)

      # ---- Lake test results
      # even if not running oneshot or instance modes, workflow records what it can, so just print what's available

      self._add_completion_summary(summary, batch)  # big table of characters and images at every stage in AMTL

      # images of comparison metrics (few shot and instance tests)
      if match_mse_key in results:
        matching_matrices = results.pop(match_mse_key)

        # remove but ignore the olap one
        if match_olap_key in results:
          results.pop(match_olap_key)

        logging.debug(matching_matrices.keys())
        accuracy_metrics = ['truth', 'dg_hidden', 'dg_recon', 'vc_recon']

        matrices = []
        present_features = []
        for feature, matrix in matching_matrices.items():
          if feature not in accuracy_metrics:
            continue
          matrices.append(matrix)
          present_features.append(feature)

        matrices_4d = np.expand_dims(np.array(matrices), axis=3)
        image_utils.arbitrary_image_summary(summary, matrices_4d,
                                            name=self._component.name + '/few-shot-summaries/' + 'matching_matrices',
                                            image_names=present_features)

      # all the metrics added to losses[] (called results here) at different stages
      for metric, metric_value in results.items():
        if metric in skip_summaries:
          continue
        summary.value.add(tag=self._component.name + '/few-shot-summaries/' + metric, simple_value=metric_value)

      self._writer.add_summary(summary, batch)
      self._writer.flush()

  def _create_data_subset(self):
    # if the overlap main modes are: mu_intra, and mu_inter, with mu_intra > mu_inter
    # and a good point in between is T, then T is threshold to separate good and bad samples Si, from the batch, B
    # THEN: only keep if:    min(overlap_intra(Si, B)) > T and max(overlap_inter(Si, B)) < T

    thresh = 60
    margin = 0

    def overlap_test_thresh(idx, vals, labels):
      overlap_intra, overlap_inter = overlap_sample_batch(idx, vals, labels, vals_test, labels_test)
      keep = (np.max(overlap_intra) > thresh + margin) and (np.max(overlap_inter) < thresh - margin)
      return keep

    def overlap_test_error(idx, vals, labels, vals_test, labels_test):
      overlap_intra, overlap_inter = overlap_sample_batch(idx, vals, labels, vals_test, labels_test)
      return np.max(overlap_intra) > np.max(overlap_inter)

    vals_train, labels_train = self._training_features['vc'], self._training_features['labels']
    vals_test, labels_test = self._testing_features['vc'], self._testing_features['labels']

    filename = './data/mnist_recordset/mnist_subset.tfrecords'
    dataset_shape = self._dataset.shape  # [-1, 28, 28, 1]
    data_utils.write_subset(filename, dataset_shape,
                            vals_train, labels_train,
                            vals_test, labels_test,
                            keep_fn=overlap_test_error)

  def _prep_for_summaries_core(self, input_images, batch_type):
    """ Add main EpisodicComponent signals to list for later use by summary """

    _, vc_input_shape = self._component.get_signal('vc_input')
    self._summary_images['vc_input_' + batch_type] = ('vc_input', input_images, vc_input_shape)

    # vc = self._component.get_vc().get_encoding()
    vc = self._component.get_vc_encoding()  # pooled and normed (if use vc direct pooling, we get the unpooled encoding)
    _, vc_shape = self._component.get_signal('vc')
    self._summary_images['vc_' + batch_type] = ('vc', vc, vc_shape)

    if self._component.get_pc() is not None:
      pc_input_direct = self._component.get_pc().get_input(batch_type)  # 'training' will be target from DG (-1, 1), 'encoding' will be PR output (0, 1)
      _, pc_input_shape = self._component.get_signal('pc_input')
      self._summary_images['pc_in_' + batch_type] = ('pc_input', pc_input_direct, pc_input_shape)

  def _prep_for_summaries_after_training(self, input_images):
    #       0. Training Input
    #       1. Training VC Output
    #       2. Training PC Input (if pc)
    self._summary_images.clear()
    self._prep_for_summaries_core(input_images, 'training')

  def _prep_for_summaries_after_completion(self, input_images, with_comparison_images=True):
    #       3. Encoding Input
    #       4. Encoding VC Output
    #       5. Encoding PC Input (if pc)
    #       6. Encoding PC Output (if pc)

    self._prep_for_summaries_core(input_images, 'encoding')

    batch_size = self._component.batch_size()
    test_labels = self._testing_features['labels']
    train_labels = self._training_features['labels']

    if with_comparison_images:
      create_and_add_comparison_image(self._summary_images, batch_size, name='vc_ovlap',
                                      train_key='vc_training', test_key='vc_encoding',
                                      use_mse=False, k=-1, train_labels=train_labels, test_labels=test_labels)

    if self._component.get_pc() is not None:

      pc = self._component.get_pc().get_decoding()
      _, pc_shape = self._component.get_signal('pc')
      self._summary_images['pc_encoding'] = ('pc', pc, pc_shape)

      if with_comparison_images:
        create_and_add_comparison_image(self._summary_images, batch_size, name='pc_in_mse',
                                        train_key='pc_in_training', test_key='pc_in_encoding',
                                        use_mse=True, k=-1, train_labels=train_labels, test_labels=test_labels)

        create_and_add_comparison_image(self._summary_images, batch_size, name='pc_in_tf_mse',
                                        train_key='pc_in_encoding', test_key='pc_in_training',
                                        use_mse=True, k=-1, train_labels=train_labels, test_labels=test_labels)

        # create_and_add_comparison_image(self._summary_images, batch_size, name='pc_out_ovlap',
        #                                 train_key='pc_in_training', test_key='pc_encoding',
        #                                 use_mse=True, k=-1, train_labels=train_labels, test_labels=test_labels)

        # create_and_add_comparison_image(self._summary_images, batch_size, name='pc_out_mse',
        #                                 train_key='pc_in_training', test_key='pc_encoding',
        #                                 use_mse=True, k=-1, train_labels=train_labels, test_labels=test_labels)

      if self._component.get_pc().use_pm_raw is True:
        ec_out_raw = self._component.get_pc().get_ec_out_raw()
        _, ec_out_raw_shape = self._component.get_signal('ec_out_raw')
        self._summary_images['ec_out_raw'] = ('ec_out_raw', ec_out_raw, ec_out_raw_shape)

      if 'pc_at_vc' in self._testing_features:
        pc_at_vc = self._testing_features['pc_at_vc']
        _, pc_at_vc_shape = self._component.get_signal('vc_input')
        self._summary_images['pc_at_vc'] = ('pc_at_vc', pc_at_vc, pc_at_vc_shape)

  def _add_completion_summary(self, summary, batch):
    """
    Show all the relevant images put in _summary_images by the _prep methods.
    They are collected during training and testing.
    """
    images_names = [
        'vc_input_training',
        # 'vc_training',
        'pc_in_training',
        'vc_input_encoding',
        # 'vc_encoding',      # not very informative
        'pc_in_encoding',
        'pc_encoding',
        'pc_at_vc'      # available when not using interest filter
    ]

    if self._hparams.pc_type != 'hl':
      images_names.remove('pc_in_training')
      images_names.remove('pc_in_encoding')
      images_names.remove('pc_encoding')

    if self._hparams.use_pm:
      images_names += ['ec_out_raw']

    if self._add_comparison_images:
      images_names += [
          'vc_ovlap',
          'pc_in_mse',
          'pc_in_tf_mse',
          'pc_out_mse'
      ]

    summary_images = []
    for image_name in images_names:   # this defines the order
      if image_name in self._summary_images:    # only try to add if it is possible
        summary_images.append(self._summary_images[image_name])

    if self._opts['summarize_completion'] != 'none':
      to_file = (self._opts['summarize_completion'] == 'to_file')
      if self._paper is True:
        add_completion_summary_paper(summary_images, self._summary_dir, batch, to_file)
      else:
        add_completion_summary(summary_images, self._summary_dir, summary, batch, to_file)

  def _compute_few_shot_metrics(self, losses, modes, pc_in, pc_completion, pc_at_dg, pc_at_vc):
    """
    After information flows through AMTL up to PC, then we decode back through the dg and vc.
    Find the best matching sample from the train batch, for each of these stages,
    by comparing with the equivalent signal (given by the 'key' in the features dic)
    i.e. compare test pc completion with the train dg encoding
    @:param losses add the metrics to this list
    @:param mode 'oneshot' or 'instance' (see hyperparams for workflow)
    """

    testing_features = {
      'labels': self._testing_features['labels'],
      'vc': self._testing_features['vc']   # hack, to test just using ff vc, with mse (instead of overlap)
    }
    if pc_completion is not None:
      testing_features['pc'] = pc_completion    # comparison with whole PC including cue_nn
      testing_features['pc_in'] = pc_in         # comparison _without_ PC (but with cue_nn)
    if pc_at_dg is not None:
      testing_features['dg_recon'] = pc_at_dg
    if pc_at_vc is not None:
      testing_features['vc_recon'] = pc_at_vc
      testing_features['pc_at_vc'] = pc_at_vc

    def add_matching_to_averages(matching_matrices_, matching_accuracies_, sum_ambiguous_, keys, prefixes):
      losses[keys[0]] = matching_matrices_
      losses[keys[1]] = matching_accuracies_
      losses[keys[2]] = sum_ambiguous_

      # record loss for the metrics - to report averages later on
      for accuracy_type, val in matching_accuracies_.items():
        if accuracy_type == 'labels':
          continue
        self._report_average_metric('{}_{}'.format(prefixes[0], accuracy_type), val)

      for ambig_type, val in sum_ambiguous_.items():
        self._report_average_metric('{}_{}'.format(prefixes[1], ambig_type), val)

    # matching matrices method (mse on a completion at different levels of hierarchy)
    matching_matrices, matching_accuracies, sum_ambiguous = compute_matching(modes, self._training_features,
                                                                             testing_features, 'mse')
    add_matching_to_averages(matching_matrices, matching_accuracies, sum_ambiguous,
                             keys=[match_mse_key, match_acc_mse_key, sum_ambiguous_mse_key],
                             prefixes=['acc_mse', 'amb_mse'])

    # matching matrices method ----> *Test vs Train (inverse of Lake)*
    matching_matrices, matching_accuracies, sum_ambiguous = compute_matching(modes, testing_features,
                                                                             self._training_features, 'mse')
    add_matching_to_averages(matching_matrices, matching_accuracies, sum_ambiguous,
                             keys=[match_mse_tf_key, match_acc_mse_tf_key, sum_ambiguous_mse_tf_key],
                             prefixes=['acc_mse_tf', 'amb_mse_tf'])

    # matching matrices method (olap on a completion at different levels of hierarchy)
    matching_matrices, matching_accuracies, sum_ambiguous = compute_matching(modes, self._training_features,
                                                                             testing_features, 'overlap')

    add_matching_to_averages(matching_matrices, matching_accuracies, sum_ambiguous,
                             keys=[match_olap_key, match_acc_olap_key, sum_ambiguous_olap_key],
                             prefixes=['acc_olap', 'amb_olap'])

    matching_matrices, matching_accuracies, sum_ambiguous = compute_matching(modes, testing_features, 
                                                                             self._training_features, 'overlap')

    add_matching_to_averages(matching_matrices, matching_accuracies, sum_ambiguous,
                             keys=[match_olap_tf_key, match_acc_olap_tf_key, sum_ambiguous_olap_tf_key],
                             prefixes=['acc_olap_tf', 'amb_olap_tf'])

  def _report_average_metric(self, name, val):
    """
    Report this metric as an average value across batches.
    It is recorded here, and printed at the end of the run.
    """

    logging.debug('report average: ' + name)

    if name not in self._total_losses.keys():
      self._total_losses[name] = []
    self._total_losses[name].append(val)

  def _extract_features(self, fetched):
    """
    dic(key=feature, val=[label 1 value, label 2 value, label 3 value, ......])
      feature can also = 'label' and values are the label names
    """
    features = {
        'labels': fetched['labels']
    }

    if self._component.is_build_dg():
      dg_hidden = self._component.get_dg().get_encoding()
      dg_recon = self._component.get_dg().get_decoding()

      features.update({
          'dg_hidden': dg_hidden,
          'dg_recon': dg_recon
      })

    vc_hidden, vc_recon = self.get_vc_encoding_decoding()
    features.update({
      'vc': vc_hidden,
      'vc_recon': vc_recon
    })

    vc_core_hidden, vc_core_recon = self.get_vc_core_encoding_decoding()
    features.update({
      'vc_core_hidden': vc_core_hidden,
      'vc_core_recon': vc_core_recon
    })

    return features

  def _add_test_dg_summaries(self, results, batch):
    bins = 200

    inter = results['inter']

    # Calc dist stats
    min_inter = np.min(inter)
    max_inter = np.max(inter)

    summary = tf_utils.histogram_summary(tag=self._component.name + '/fs/dg_inter', values=inter,
                                         bins=bins, minimum=min_inter, maximum=max_inter)
    self._writer.add_summary(summary, batch)

    return summary

  def _add_test_vc_summaries(self, results, batch, scope="", fig_scope=""):
    bins = 200
    num_sd = 4.0  # How much spread visible

    if scope != "":
      scope = scope + "_"

    # overlap comparison
    inter = results[scope + 'inter']  # important to pop so that we can add the rest to the summaries easily
    intra = results[scope + 'intra']

    inter_per_label = results.get(scope + 'inter_per_label', {})
    intra_per_label = results.get(scope + 'intra_per_label', {})

    dbug = False
    if dbug:
      # Calc dist stats
      min_inter = np.min(inter)
      max_inter = np.max(inter)
      min_intra = np.min(intra)
      max_intra = np.max(intra)

      print("\n--------- VC Overlap Debugging ----------")
      print("min inter = " + str(min_inter))
      print("min intra = " + str(min_intra))
      print("max inter = " + str(max_inter))
      print("max intra = " + str(max_intra))

    sd_inter = np.std(inter)
    sd_intra = np.std(intra)
    mean_inter = np.mean(inter)
    mean_intra = np.mean(intra)

    # clip at N SD from mean
    min_inter = max(0, mean_inter - num_sd * sd_inter)
    max_inter = mean_inter + num_sd * sd_inter
    min_intra = max(0, mean_intra - num_sd * sd_intra)
    max_intra = mean_intra + num_sd * sd_intra

    # align axes
    align_axes = True
    if align_axes:
      min_inter = min(min_inter, min_intra)
      min_intra = min_inter
      max_inter = max(max_inter, max_intra)
      max_intra = max_inter

    inter_size = len(inter)
    intra_size = len(intra)

    # clip distributions
    inter = np.clip(inter, min_inter, max_inter)
    intra = np.clip(intra, min_intra, max_intra)

    inter_size_removed = inter_size - len(inter)
    intra_size_removed = intra_size - len(intra)

    if dbug:
      print("\n---- histograms ----")
      print("min inter = " + str(min_inter))
      print("min intra = " + str(min_intra))
      print("max inter = " + str(max_inter))
      print("max intra = " + str(max_intra))
      print("Clipped inter={0}, intra={1} points.".format(inter_size_removed, intra_size_removed))

    summary = tf_utils.histogram_summary(tag=self._component.name + '/fs/' + fig_scope + 'vc_inter', values=inter,
                                         bins=bins, minimum=min_inter, maximum=max_inter)
    self._writer.add_summary(summary, batch)

    summary = tf_utils.histogram_summary(tag=self._component.name + '/fs/' + fig_scope + 'vc_intra', values=intra,
                                         bins=bins, minimum=min_intra, maximum=max_intra)
    self._writer.add_summary(summary, batch)

    for label, inter_matrix in inter_per_label.items():
      summary = tf_utils.histogram_summary(tag=self._component.name + '/fs/' + fig_scope + 'vc_inter_per_label/' + str(label),
                                           values=inter_matrix, bins=bins, minimum=min_inter, maximum=max_inter)
      self._writer.add_summary(summary, batch)

    for label, intra_matrix in intra_per_label.items():
      summary = tf_utils.histogram_summary(tag=self._component.name+'/fs/'+fig_scope+'vc_intra_per_label/'+str(label),
                                           values=intra_matrix, bins=bins, minimum=min_intra, maximum=max_intra)
      self._writer.add_summary(summary, batch)

    # and accuracy margin
    accuracy_margin = results[scope + 'accuracy_margin']
    mn = np.min(accuracy_margin)
    mx = np.max(accuracy_margin)
    summary = tf_utils.histogram_summary(tag=self._component.name + '/fs/' + fig_scope + 'vc_acc_margin',
                                         values=accuracy_margin, bins=bins, minimum=mn, maximum=mx)
    self._writer.add_summary(summary, batch)

    return summary

  def get_vc_core_encoding_decoding(self):
    """Get the info of the vc subcomponent itself, not the vc section which can include Interest Filtering etc."""
    if EpisodicComponent.is_vc_hierarchical():
      encoding = self._component.get_vc().get_encoding('vc0')
      decoding = self._component.get_vc().get_decoding('vc0')
    else:
      encoding = self._component.get_vc().get_encoding()
      decoding = self._component.get_vc().get_decoding()

    return encoding, decoding

  def get_vc_encoding_decoding(self):
    """
    Get the output of the entire vc section (not just the vc component)
    i.e. after vc() there is an optional InterestFilter, pooling and potentially other conditioning
    """

    # whichever vc is used internally, before it moves to the next stages, this is the encoding
    encoding = self._component.get_vc_encoding()

    if EpisodicComponent.is_vc_hierarchical():
      decoding = self._component.get_vc().get_decoding('vc0')
    else:
      decoding = self._component.get_vc().get_decoding()

    return encoding, decoding

  def _reinitialize_networks(self):
    if not self._component.is_build_pc():
      return

    outer_scope = self._component.name + '/' + self._component.get_pc().name
    variables = self._component.get_pc().variables_networks(outer_scope)

    if self._component.is_build_ll_pc():
      outer_scope = self._component.name + '/' + self._component.get_ll_pc().name
      variables += self._component.get_ll_pc().variables_networks(outer_scope)

    logging.info('Reinitialise PC network variables: {0}'.format(variables))

    init_nets = tf.variables_initializer(variables)
    self._session.run(init_nets)
