# Copyright (C) 2018 Project AGI
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

"""LabelLearnerFC class."""

import logging

import numpy as np
import tensorflow as tf

from pagi.components.summarize_levels import SummarizeLevels
from pagi.utils import image_utils
from pagi.utils.layer_utils import type_activation_fn

from pagi.components.summary_component import SummaryComponent

class LabelLearnerFC(SummaryComponent):

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        batch_size=1,
        learning_rate=0.0001,
        optimizer='adam',
        train_input_dropout_keep_prob=1.0,  # 1.0=off
        train_hidden_dropout_keep_prob=1.0,  # 1.0=off
        test_with_noise=0.0,  # 0.0=off
        train_with_noise=0.0,  # 0.0=off
        train_with_noise_pp=0.0,
        test_with_noise_pp=0.0,
        hidden_size=500,
        non_linearity='leaky_relu',
        l2_regularizer=0.0,
        summarize_level=SummarizeLevels.ALL.value,
        max_outputs=3
    )

  def get_batch_type(self):
    return self._batch_type

  def _kernel_initializer(self):
    w_factor = 1.0  # factor=1.0 for Xavier, 2.0 for He
    # w_mode = 'FAN_IN'
    w_mode = 'FAN_AVG'
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=w_factor, mode=w_mode,
                                                                        uniform=False)
    return kernel_initializer

  def build(self, target_output, train_input, test_input, hparams, name='ll'):
    """Build the Label Learner network."""
    self.name = name
    self._hparams = hparams

    with tf.variable_scope(self.name):
      self._batch_type = tf.placeholder_with_default(input='training', shape=[], name='batch_type')

      # 0) nn params
      # ------------------------------------
      non_linearity = self._hparams.non_linearity
      hidden_size = self._hparams.hidden_size

      # 1) organise inputs to network
      # ------------------------------------
      self._dual.set_op('target_output', target_output)
      t_nn_shape = target_output.get_shape().as_list()
      t_nn_size = np.prod(t_nn_shape[1:])

      x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                     lambda: train_input,
                     lambda: test_input)

      x_nn = tf.layers.flatten(x_nn)

      # 2) build the network
      # ------------------------------------
      # apply noise at train and/or test time, to regularise / test generalisation
      x_nn = tf.cond(tf.equal(self._batch_type, 'encoding'),
                     lambda: image_utils.add_image_salt_noise_flat(x_nn, None,
                                                                   noise_val=self._hparams.test_with_noise,
                                                                   noise_factor=self._hparams.test_with_noise_pp),
                     lambda: x_nn)

      x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                     lambda: image_utils.add_image_salt_noise_flat(x_nn, None,
                                                                   noise_val=self._hparams.train_with_noise,
                                                                   noise_factor=self._hparams.train_with_noise_pp),
                     lambda: x_nn)

      # apply dropout during training
      input_keep_prob = self._hparams.train_input_dropout_keep_prob
      x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                     lambda: tf.nn.dropout(x_nn, input_keep_prob),
                     lambda: x_nn)

      self._dual.set_op('x_nn', x_nn)

      x_nn = tf.layers.flatten(x_nn)

      # Hidden layer[s]
      weights = []

      # Build hidden layer(s)
      if hidden_size > 0:
        layer_hidden = tf.layers.Dense(units=hidden_size,
                                       activation=type_activation_fn(non_linearity),
                                       name='hidden',
                                       kernel_initializer=self._kernel_initializer())

        hidden_out = layer_hidden(x_nn)
        weights.append(layer_hidden.weights[0])
        weights.append(layer_hidden.weights[1])

        hidden_keep_prob = self._hparams.train_hidden_dropout_keep_prob
        hidden_out = tf.cond(tf.equal(self._batch_type, 'training'),
                             lambda: tf.nn.dropout(hidden_out, hidden_keep_prob),
                             lambda: hidden_out)
      else:
        hidden_out = x_nn

      # Build output layer
      layer_out = tf.layers.Dense(units=t_nn_size,
                                  name='logits',
                                  kernel_initializer=self._kernel_initializer())

      logits = layer_out(hidden_out)
      weights.append(layer_out.weights[0])
      weights.append(layer_out.weights[1])

      f = tf.nn.softmax(logits)  # Unit range
      y = tf.stop_gradient(f)

      self._dual.set_op('preds', y)
      self._dual.set_op('logits', logits)

      # Compute accuracy
      preds = self._dual.get_op('preds')
      labels = self._dual.get_op('target_output')

      unseen_idxs = (0, 1)

      correct_predictions = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
      correct_predictions = tf.cast(correct_predictions, tf.float32)

      self._dual.set_op('correct_predictions', correct_predictions)
      self._dual.set_op('accuracy', tf.reduce_mean(correct_predictions))
      self._dual.set_op('accuracy_unseen', tf.reduce_mean(correct_predictions[unseen_idxs[0]:unseen_idxs[1]]))
      self._dual.set_op('total_correct_predictions', tf.reduce_sum(correct_predictions))

      # Build loss function
      loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_output, logits=logits)
      loss = tf.reduce_mean(loss)
      self._dual.set_op('loss', loss)

      if self._hparams.l2_regularizer > 0.0:
        all_losses = [loss]

        for weight in weights:
          weight_loss = tf.nn.l2_loss(weight)
          weight_loss_sum = tf.reduce_sum(weight_loss)
          weight_loss_scaled = weight_loss_sum * self._hparams.l2_regularizer
          all_losses.append(weight_loss_scaled)

        all_losses_op = tf.add_n(all_losses)
        self._build_optimizer(all_losses_op, 'training_ll', scope=name)
      else:
        self._build_optimizer(loss, 'training_ll', scope=name)

      return y

  def _build_optimizer(self, loss_op, training_op_name, scope=None):
    """Minimise loss using initialised a tf.train.Optimizer."""

    logging.info("-----------> Adding optimiser for op %s", loss_op)

    if scope is not None:
      scope = 'optimizer/' + str(scope)
    else:
      scope = 'optimizer'

    with tf.variable_scope(scope):
      optimizer = self._setup_optimizer()
      training = optimizer.minimize(loss_op, global_step=tf.train.get_or_create_global_step())

      self._dual.set_op(training_op_name, training)

  def _setup_optimizer(self):
    """Initialise the Optimizer class specified by a hyperparameter."""
    if self._hparams.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self._hparams.learning_rate)
    elif self._hparams.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(self._hparams.learning_rate, self._hparams.momentum,
                                             use_nesterov=self._hparams.momentum_nesterov)
    elif self._hparams.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self._hparams.learning_rate)
    else:
      raise NotImplementedError('Optimizer not implemented: ' + str(self._hparams.optimizer))

    return optimizer

  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Updates the feed dict."""

    names = []
    self._dual.update_feed_dict(feed_dict, names)

    feed_dict.update({
        self._batch_type: batch_type
    })

  def add_fetches(self, fetches, batch_type='training'):
    """Adds ops that will get evaluated."""
    names = ['loss', 'accuracy', 'accuracy_unseen', 'target_output', 'preds']

    if batch_type == 'training':
      names.extend(['training_ll'])

    self._dual.add_fetches(fetches, names)

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""

    # Loss (not a tensor)
    self._loss = fetched[self.name]['loss']

    names = ['loss', 'accuracy', 'accuracy_unseen', 'target_output', 'preds']
    self._dual.set_fetches(fetched, names)

    # Summaries
    super().set_fetches(fetched, batch_type)

  def _build_summaries(self, batch_type=None, max_outputs=3):
    """Builds all summaries."""
    summaries = []
    max_outputs = self._hparams.max_outputs

    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    correct_predictions_summary = tf.summary.scalar('correct_predictions',
                                                    self._dual.get_op('total_correct_predictions'))
    summaries.append(correct_predictions_summary)

    accuracy_summary = tf.summary.scalar('accuracy', self._dual.get_op('accuracy'))
    summaries.append(accuracy_summary)

    accuracy_unseen_summary = tf.summary.scalar('accuracy_unseen', self._dual.get_op('accuracy_unseen'))
    summaries.append(accuracy_unseen_summary)

    # Loss
    loss_summary = tf.summary.scalar('loss', self._dual.get_op('loss'))
    summaries.append(loss_summary)

    return summaries

  def variables_networks(self, outer_scope):
    vars_nets = []

    # Selectively include/exclude optimizer parameters
    optim_ll = False

    vars_nets += self._variables_ll(outer_scope)

    if optim_ll:
      vars_nets += self._variables_ll_optimizer(outer_scope)

    return vars_nets

  @staticmethod
  def _variables_ll(outer_scope):
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/hidden"
    )
    variables += tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/logits"
    )
    return variables

  @staticmethod
  def _variables_ll_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer"
    )
