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

"""PatternCompletionWorkflow Workflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from enum import Enum

import numpy as np
import tensorflow as tf

from pagi.workflows.workflow import Workflow
from pagi.utils import logger_utils, image_utils, np_utils
from pagi.utils.generic_utils import class_filter
from pagi.utils.tf_utils import tf_label_filter, tf_invert, tf_set_min

from pagi.datasets.omniglot_dataset import OmniglotDataset
from aha.datasets.omniglot_lake_dataset import OmniglotLakeDataset
from aha.datasets.omniglot_lake_runs_dataset import OmniglotLakeRunsDataset
from aha.utils.recursive_component_harness import RecursiveComponentHarness


class UseTrainForTest(Enum):
  IDENTICAL = 1
  SHUFFLED = 2
  NO = 3


class PatternCompletionWorkflow(Workflow):
  """Pattern completion experiment workflow."""

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):
    super().__init__(session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
                     opts=opts, summarize=summarize, seed=seed, summary_dir=summary_dir,
                     checkpoint_opts=checkpoint_opts)
    self._rsummary_from_batch = 0
    self._recursive_harness = None
    self._input_mode = None

    # these are for summaries
    self._memorised = None
    self._cue = None  # cue refers to input presented to Hopfield (not the cue internal to Hop)
    self._recalled = None

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        num_repeats=1,
        invert_images=False,
        min_val=0,              # set any 0 in the input image, to this new min_val. ---> if ==0, then don't do anything.
        train_classes=['5', '6', '7', '8', '9'],
        test_classes=['5', '6', '7', '8', '9'],
        batch_all_classes=False,    # Ensure the batch has at least one example of each of the specified classes
        batch_no_duplicates=False,  # Ensure the batch ONLY one example of each of the specified classes
        completion_gain=1.0,
        train_recurse=False,
        test_recurse=False,
        recurse_iterations=0,
        rsummary_batches=2,     # number of batches at the end, to write recursive summaries i.e. 2, then last 2 batches
        degrade_type='horizontal',  # none, vertical, horizontal or random (the model completes image degraded)
        degrade_factor=0.5,     # fraction to be degraded, if supported by the degrade_type option
        degrade_value=0.0,      # when degrading pixels, set them to this value
        noise_val=1.0,          # value of 'active' bits of salt and pepper noise
        noise_factor=0.2,       # fraction of image to be corrupted with noise
        input_mode={
            "train_first": "complete",
            "train_inference": "complete",
            "test_first": "complete",
            "test_inference": "complete"
        },
        evaluate=True,
        train=True
    )

  #######################################################
  # Utility methods used by training() and evaluate()

  def _is_inference_recursive(self):
    test_recurse = self._opts['evaluate'] and \
                   self._opts['test_recurse']
    return test_recurse

  def _is_train_recursive(self):
    train_recurse = self._opts['train'] and \
                    self._opts['train_recurse']
    return train_recurse

  def _is_combined_train_and_test_recursive_summary(self):
    """
    In the separate recursive summary, should we include
    training and testing iterations into one continuous plot.
    """
    if self._is_inference_recursive() and self._is_train_recursive():
      return True
    return False

  def _is_write_evaluate_summary(self):
    """If we should write a summary at the end of evaluate, independent of recursive summary."""
    return True

  #######################################################

  def _is_omniglot_lake(self):
    return self._dataset_type.__name__.startswith('Omniglot')
    # return (self._dataset_type.__name__ == OmniglotDataset.__name__) or (
    #     self._dataset_type.__name__ == OmniglotLakeDataset.__name__) or (
    #     self._dataset_type.__name__ == OmniglotLakeRunsDataset.__name__)

  @staticmethod
  def validate_batch_include_all_classes(feature, label, classes):  # pylint: disable=W0613
    """Ensures the batch has the specified classes."""
    classes = tf.convert_to_tensor(classes)
    unique_labels, _ = tf.unique(label)
    return tf.equal(tf.reduce_sum(classes), tf.reduce_sum(unique_labels))

  def validate_batch_no_duplicate_classes(self, feature, label, classes):  # pylint: disable=W0613
    """Ensures the batch has no more than one each of the specified classes."""
    unique_labels, _ = tf.unique(label)
    batch_size = self._hparams.batch_size
    return tf.equal(tf.size(unique_labels), batch_size)

  def _gen_datasets_with_options(self, train_classes=None, test_classes=None, is_superclass=None, class_proportion=0.0,
                                 degrade_test=False, degrade_type='random', degrade_val=0, degrade_factor=0.5,
                                 noise_test=False, noise_type='sp_binary', noise_factor=0.2,
                                 recurse_train=False, recurse_test=False,
                                 num_batch_repeats=1, recurse_iterations=1, additional_test_decodes=0,
                                 evaluate_step=True, use_trainset_for_tests=UseTrainForTest.IDENTICAL,
                                 invert_images=False, min_val=0):
    """
    Generate degraded datasets for pattern completion.

    The test is a duplication of the train set, but degraded versions.
    Each batch of the train set is then duplicated (in place), so it is in pairs.
    During operation, the first of this pair is used for training, the second for comparison with test batch in eval.

    Repeat each batch 'num_repeats' times.
    This is so that you can repeatedly train on the exact same batch (for episodic work).

    Add an item for every set an episode, add one to the train set, b/c it is taken for comparison in test step.

    E.g.If the stream of batches is    A, B, C, D.
    Iterations = 3, num_repeats = 2

    Without any additional decodes in evaluate()
      Train = [[A, A, A], A], [[A, A, A], A], [[B, B, B], B], [[B, B, B], B],
      Test  = [A, A, A], [A, A, A], [B, B, B], [B, B, B],

    With 2 additional decodes in evaluate() e.g. vc_decode() and dg_decode()
      Train = [[A, A, A], A], [[A, A, A], A], [[B, B, B], B], [[B, B, B], B],
      Test  = [[A, A, A], A, A], [[A, A, A], A, A], [[B, B, B], B, B], [[B, B, B], B, B],

    :param train_classes: a list specifying the classes for training, all if None
    :param test_classes: a list specifying the classes for testing, all if None
    :param is_superclass: if true, class labels refer to superclasses
    :param class_proportion: for filtering superclass
    :param degrade_test: optionally degrade test images
    :param degrade_type: if degrading, use this method
    :param degrade_val: set 0's to this val
    :param degrade_factor: fraction to be degraded (if the degrade type supports that option)
    :param recurse_train: recursion used for training
    :param recurse_test: recursion used for inference
    :param num_batch_repeats: present each batch this many times
    :param recurse_iterations: number of iterations for each batch (>1 if recursion)
    :param evaluate_step: true if there is an evaluation after the train step (then extra train image taken each step)
    :param invert_images: white->black and black->white (assumes image is [0,1])
    :param use_trainset_for_tests: UseTrainForTest: identical, shuffled, no
    :return:

    """

    def get_set(set_type, for_evaluate, seed_increment=0):
      """
      'set_type' = 'train', or assumed it is Test set type
      'for_evaluate', boolean, when train set is used for the evaluate phase
      'seed_increment' is for the case where you are using the same dataset for train and test,
      but you want to fetch different exemplars


      --------> VERY IMPORTANT: In this version, it assumes that additional things (recursion and decodes)
                                are done at the end of num_batch_repeats iterations
                                In previous versions or other workflows, it may expect this EVERY iteration

      """

      is_train = set_type == 'train'

      repeats = num_batch_repeats

      if recurse_train and is_train:
        repeats = repeats * recurse_iterations

      if recurse_test and (not is_train or for_evaluate):
        repeats = repeats + (recurse_iterations-1)    # only do recurse_iterations once per 'num_batch_repeats'. -1 because recurse iterations of 1 means 1 normal iteration, none additional

      if evaluate_step:
        if is_train and not for_evaluate:
          # another 'train' input for every batch (num_batch_repeats), will be iterated for comprsn `complete_pattern()`
          repeats = repeats + num_batch_repeats
        else:
          # additional decodes for every every num_batch_repeats
          repeats = repeats + additional_test_decodes

      logging.debug("-----------------------> is_train={},   repeat={}   ({})".format(is_train, repeats, additional_test_decodes))

      if is_train:
        the_dataset = self._dataset.get_train()
      else:
        the_dataset = self._dataset.get_test()

      if not self._is_omniglot_lake():
        if is_train:
          the_classes = train_classes
        else:
          the_classes = test_classes

        # Filter dataset to keep specified classes only
        if the_classes and len(the_classes) > 0:
          the_classes = class_filter(self._dataset, the_classes, is_superclass, class_proportion)
          the_dataset = the_dataset.filter(lambda x, y: tf_label_filter(x, y, the_classes))

        the_dataset = the_dataset.shuffle(buffer_size=10000, seed=(self._seed+seed_increment))

      if self._opts['evaluate_mode'][0] == 'simple' and self._opts['evaluate_mode'][1].startswith('run'):
        the_dataset = the_dataset.shuffle(buffer_size=10000, seed=(self._seed+seed_increment))

      if invert_images:
        the_dataset = the_dataset.map(lambda x, y: tf_invert(x, y))

      if min_val != 0:
        the_dataset = the_dataset.map(lambda x, y: tf_set_min(x, y, min_val))

      the_dataset = the_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._hparams.batch_size))

      if not self._is_omniglot_lake():
        # Ensure the batch has at least one example of each of the specified class
        if self._opts['batch_all_classes']:
          the_dataset = the_dataset.filter(lambda x, y: self.validate_batch_include_all_classes(x, y, train_classes))

        # Ensure the batch ONLY one example of each of the specified class
        if self._opts['batch_no_duplicates']:
          the_dataset = the_dataset.filter(lambda x, y: self.validate_batch_no_duplicate_classes(x, y, train_classes))

      # Optionally degrade input image
      # TODO just for now specify random value so that you don't get variation within the recursive iterations
      if for_evaluate and degrade_test:
        the_dataset = the_dataset.map(lambda x, y: image_utils.degrade_image(x, y,
                                                                             degrade_type,
                                                                             degrade_val,
                                                                             degrade_factor=degrade_factor,
                                                                             random_value=0.3))
      if for_evaluate and noise_test:
        the_dataset = the_dataset.map(lambda x, y: image_utils.add_image_noise(x, y,
                                                                               minval=min_val,
                                                                               noise_type=noise_type,
                                                                               noise_factor=noise_factor))

      logging.debug("isTrain, for_evaluate, repeats: ", is_train, for_evaluate, repeats)

      the_dataset = the_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(repeats))
      the_dataset = the_dataset.prefetch(1)
      the_dataset = the_dataset.repeat()

      return the_dataset

    train_dataset = get_set(set_type='train', for_evaluate=False)

    test_dataset = None
    if evaluate_step:
      if use_trainset_for_tests == UseTrainForTest.SHUFFLED:
        test_dataset = get_set(set_type='train', for_evaluate=True, seed_increment=1)
      elif use_trainset_for_tests == UseTrainForTest.IDENTICAL:
        test_dataset = get_set(set_type='train', for_evaluate=True)
      else:  # NO
        test_dataset = get_set(set_type='test', for_evaluate=True, seed_increment=2)  # use a different seed in case train and test are the same as in artificial or recordset dataset

    return train_dataset, test_dataset

  def _generate_datasets(self):
    """Override in child classes with your options"""

    degrate_test = False
    if self._opts['degrade_type'] != 'none':
      degrate_test = True

    noise_test = False
    if self._opts['noise_val'] != 0:
      noise_test = True

    train_dataset, test_dataset = self._gen_datasets_with_options(self._opts['train_classes'],
                                                                  self._opts['test_classes'],
                                                                  degrade_test=degrate_test,
                                                                  degrade_type=self._opts['degrade_type'],
                                                                  degrade_factor=self._opts['degrade_factor'],
                                                                  degrade_val=self._opts['min_val'],
                                                                  noise_test=noise_test,
                                                                  noise_factor=self._opts['noise_factor'],
                                                                  recurse_train=self._is_train_recursive(),
                                                                  recurse_test=self._is_inference_recursive(),
                                                                  num_batch_repeats=self._opts['num_repeats'],
                                                                  recurse_iterations=self._opts['recurse_iterations'],
                                                                  evaluate_step=self._opts['evaluate'],
                                                                  invert_images=self._opts['invert_images'],
                                                                  min_val=self._opts['min_val'])
    return train_dataset, test_dataset

  def _create_dataset(self):
    if self._is_omniglot_lake():
      self._dataset = self._dataset_type(self._dataset_location,
                                         self._hparams.batch_size,
                                         self._opts['test_classes'])
    else:
      self._dataset = self._dataset_type(self._dataset_location)

  def _setup_dataset(self):
    """Setup the dataset and retrieve inputs, labels and initializers"""
    with tf.variable_scope('dataset'):

      self._create_dataset()

      resize_factor = self._opts['resize_images_factor']
      if resize_factor != 1.0:
        if not self._is_omniglot_lake():
          raise RuntimeError('Resize images is only supported for Omniglot currently')
        # the dataset shape is referenced in other places, so we need to change it directly
        # that is only supported by OmniglotDataset. That is why we don't resize here outside of Dataset class.
        # Omni uses it to resize internally and then image size is consistent with dataset.dataset_shape
        height = int(resize_factor * self._dataset.shape[1])
        width = int(resize_factor * self._dataset.shape[2])
        self._dataset.set_shape(height, width)

      # Setup dataset iterators
      train_dataset, test_dataset = self._generate_datasets()

      self._placeholders['dataset_handle'] = tf.placeholder(tf.string, shape=[], name='dataset_handle')

      # Setup dataset iterators
      with tf.variable_scope('dataset_iterators'):
        self._iterator = tf.data.Iterator.from_string_handle(self._placeholders['dataset_handle'],
                                                             train_dataset.output_types,
                                                             train_dataset.output_shapes)
        self._inputs, self._labels = self._iterator.get_next()

        self._dataset_iterators = {}

        with tf.variable_scope('train_dataset'):
          self._dataset_iterators['training'] = train_dataset.make_initializable_iterator()

        if self._opts['evaluate']:
          with tf.variable_scope('test_dataset'):
            self._dataset_iterators['test'] = test_dataset.make_initializable_iterator()

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

  def _on_after_evaluate(self, results, batch):
    """Record losses after evaluation is completed."""
    for loss, loss_value in results.items():
      logger_utils.log_metrics({loss: loss_value})

    if self._summarize:
      summary = tf.Summary()
      for loss, loss_value in results.items():
        summary.value.add(tag=self._component.name + '/summaries/completion/' + loss,
                          simple_value=loss_value)

        self._add_completion_summary(summary, batch)

      self._writer.add_summary(summary, batch)
      self._writer.flush()

  def _add_completion_summary(self, summary, batch):
    """Assumes images are in appropriate shape for summary"""

    diff1 = np.abs(self._memorised - self._cue)
    diff2 = np.abs(self._memorised - self._recalled)
    mem_cue_diff = [self._memorised, self._cue, diff1]
    mem_out_diff = [self._memorised, self._recalled, diff2]

    image_utils.add_arbitrary_images_summary(summary, 'pcw', mem_cue_diff,
                                             ['memorised', 'cue', 'diff'], combined=True)
    image_utils.add_arbitrary_images_summary(summary, 'pcw', mem_out_diff,
                                             ['memorised', 'output', 'diff'], combined=True)

    np_utils.print_simple_stats(self._memorised, 'memorised')
    np_utils.print_simple_stats(self._cue, 'cue')
    np_utils.print_simple_stats(diff1, 'mem_cue_diff')
    np_utils.print_simple_stats(diff2, 'mem_out_diff')

  def _setup_recursive_train_modes(self, batch_type):
    """
    Set the appropriate mode depending on batch type.
    This is only called when recursive training
    """
    mode = 'training'
    if batch_type == 'encoding':
      mode = 'inference'
    return mode

  def training(self, training_handle, training_step, training_fetches=None):
    """The training procedure within the batch loop"""

    if training_fetches is None:
      training_fetches = {}

    batch_type = self._setup_train_batch_types()
    feed_dict = self._setup_train_feed_dict(batch_type, training_handle)

    self._component.reset()  # reset (important for feedback to be zero)

    if self._is_train_recursive():
      iterations = self._opts['recurse_iterations']
      # if setup for recursive operation, wrap in harness and step through the harness
      if self._recursive_harness is None:
        self._recursive_harness = RecursiveComponentHarness(self._component, recurse_iterations=iterations)

      mode = self._setup_recursive_train_modes(batch_type)

      summary_step = training_step - self._rsummary_from_batch  # make start index 0

      if self._is_combined_train_and_test_recursive_summary():
        summary_step = summary_step * 2   # because with 'evaluate', train and completion are concatenated

      fetched = self._recursive_harness.step(self._session,
                                             self._writer,
                                             summary_step,
                                             feed_dict=feed_dict,
                                             mode=mode,
                                             input_mode=self._input_mode,
                                             inference_batch_type=batch_type,
                                             train_batch_type=batch_type,
                                             fetches=training_fetches)
    else:
      fetched = self.step_graph(self._component, feed_dict, batch_type, training_fetches)

    # write one summary for each step of training (whether it is one graph step, or multiple recursive iterations)
    self._component.write_summaries(training_step, self._writer, batch_type=batch_type)

    return feed_dict, fetched

  def _get_target_switch_to_test(self, feed_dict, training_handle, test_handle):
    """Evaluate the target input for concrete values, then switch to the test dataset."""
    # Evaluate the input
    target_inputs = self._session.run(self._inputs, feed_dict={
        self._placeholders['dataset_handle']: training_handle
    })

    # Switch to test set
    feed_dict.update({
        self._placeholders['dataset_handle']: test_handle
    })

    return target_inputs, feed_dict

  def _set_decode_gain(self, feed_dict):
    if self._component.get_dual().get('decode_gain') is not None:
      completion_gain = self._opts['completion_gain']
      decode_gain_pl = self._component.get_dual().get('decode_gain').get_pl()
      feed_dict.update({
          decode_gain_pl: [completion_gain]
      })

  def _complete_pattern(self, feed_dict, training_handle, test_handle, test_step):
    """Exposes component to an incomplete pattern and calculates the completion loss."""
    losses = {}

    target_inputs, feed_dict = self._get_target_switch_to_test(feed_dict, training_handle, test_handle)
    self._set_decode_gain(feed_dict)
    self._inference(test_step, feed_dict)

    # Calculate pattern completion loss
    decoded_inputs = self._component.get_decoding()
    losses['completion_loss'] = np.square(abs(target_inputs - decoded_inputs)).mean()

    self._prep_for_summaries(target_inputs, self._component.get_input('encoding'), decoded_inputs)

    return losses

  def _prep_for_summaries(self, pc_memorise, pc_direct_input, pc_retrieved):
    """
    pc_memorise = the pattern to be memorised (input in 'learning' mode)
    pc_cue = the cue that was presented to PC (the input in 'retrieve' mode),
             get it from PC as it may have logic to determine which external signal
             it uses as a cue (in the case of Hopfield, it could be x_ext or from z=w.x_cue)
    pc_retrieved = the memory retrieved from PC (the output)
    """

    if len(pc_memorise.shape) == 2:   # batch of 1d vectors (other workflows, PC takes input from SAE encoding)
      shape, _ = image_utils.square_image_shape_from_1d(pc_memorise.shape[1])
    else:
      shape = pc_memorise.shape

    self._memorised = np.reshape(pc_memorise, shape)
    self._cue = np.reshape(pc_direct_input, shape)
    self._recalled = np.reshape(pc_retrieved, shape)

  def _inference(self, test_step, feed_dict, testing_fetches=None, batch_type='encoding'):
    """
    End-to-end inference for pattern completion.
    """
    if testing_fetches is None:
      testing_fetches = {}

    if self._is_inference_recursive():
      if self._recursive_harness is None:
        self._recursive_harness = RecursiveComponentHarness(self._component, self._opts['recurse_iterations'])

      summary_step = test_step - self._rsummary_from_batch  # make start index 0
      if self._is_combined_train_and_test_recursive_summary():
        summary_step = summary_step * 2 + 1   # x2 b/c train+completion, +1 so completion is after train iterations

      fetched = self._recursive_harness.step(self._session,
                                             self._writer,
                                             summary_step,
                                             feed_dict=feed_dict,
                                             mode='inference',
                                             input_mode=self._input_mode,
                                             inference_batch_type=batch_type,
                                             fetches=testing_fetches)
    else:
      fetched = self.step_graph(self._component, feed_dict, batch_type, fetches=testing_fetches)

    if self._is_write_evaluate_summary():
      self._component.write_summaries(test_step, self._writer, batch_type=batch_type)

    return fetched
