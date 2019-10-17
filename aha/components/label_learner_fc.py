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
        ll_train_dropout_keep_prob=1.0,  # 1.0=off: dropout in nn that learns the cue from EC
        test_with_noise=0.0,  # 0.0=off: noise to EC for testing generalisation of learning cue with nn
        train_with_noise=0.0,  # 0.0=off: noise to EC for testing generalisation of learning cue with nn
        train_with_noise_pp=0.0,
        test_with_noise_pp=0.0,
        hidden_size=500,
        non_linearity='leaky_relu',
        l2_regularizer=0.0,
        summarize_level=SummarizeLevels.ALL.value,
        max_outputs=3
    )

  def __init__(self):
    self._name = None
    self._hparams = None
    self._dual = None
    self._summary_values = None
    self._batch_type = None

  def build(self, target_output, train_input, test_input, name='ll'):
    """Build the Label Learner network."""

    with tf.variable_scope(self._name):
      # 0) nn params
      # ------------------------------------
      non_linearity = self._hparams.ll_linearity
      hidden_size = self._hparams.ll_hidden_size

      # 1) organise inputs to network
      # ------------------------------------

      t_nn_shape = target_output.get_shape().as_list()
      t_nn_size = np.prod(t_nn_shape[1:])

      # 2) build the network
      # ------------------------------------
      # apply noise at train and/or test time, to regularise / test generalisation
      x_nn = tf.cond(tf.equal(self._batch_type, 'encoding'),
                     lambda: image_utils.add_image_salt_noise_flat(test_input, None,
                                                                   noise_val=self._hparams.ll_test_with_noise,
                                                                   noise_factor=self._hparams.ll_test_with_noise_pp),
                     lambda: test_input)

      x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                     lambda: image_utils.add_image_salt_noise_flat(train_input, None,
                                                                   noise_val=self._hparams.ll_train_with_noise,
                                                                   noise_factor=self._hparams.ll_train_with_noise_pp),
                     lambda: train_input)

      # apply dropout during training
      keep_prob = self._hparams.ll_train_dropout_keep_prob
      x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                     lambda: tf.nn.dropout(x_nn, keep_prob),
                     lambda: x_nn)

      self._dual.set_op('ll_x_nn', x_nn)  # input to the pr path nn

      # Hidden layer[s]
      weights = []

      if hidden_size > 0:
        w_factor = 1.0  # factor=1.0 for Xavier, 2.0 for He
        # w_mode = 'FAN_IN'
        w_mode = 'FAN_AVG'
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=w_factor, mode=w_mode,
                                                                            uniform=False)

        layer_hidden = tf.layers.Dense(units=hidden_size,
                                       activation=type_activation_fn(non_linearity),
                                       name="hidden",
                                       kernel_initializer=kernel_initializer)

        hidden_out = layer_hidden(x_nn)
        weights.append(layer_hidden.weights[0])
        weights.append(layer_hidden.weights[1])

        # Optional dropout on hidden layer in addition to input dropout, potentially at different rate.
        keep_prob = 1.0
        if keep_prob < 1.0:
          hidden_out = tf.cond(tf.equal(self._batch_type, 'training'),
                               lambda: tf.nn.dropout(hidden_out, keep_prob),
                               lambda: hidden_out)
      else:
        hidden_out = x_nn

      w_factor = 1.0  # factor=1.0 for Xavier, 2.0 for He
      # w_mode = 'FAN_IN'
      w_mode = 'FAN_AVG'
      kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=w_factor, mode=w_mode,
                                                                          uniform=False)

      layer_out = tf.layers.Dense(units=t_nn_size,
                                  name="logits",
                                  kernel_initializer=kernel_initializer)  # units = number of logits
      logits = layer_out(hidden_out)

      weights.append(layer_out.weights[0])
      weights.append(layer_out.weights[1])

      f = tf.nn.softmax(logits)  # Unit range
      loss = tf.losses.softmax_cross_entropy_with_logits(target_output, logits)

      y = tf.stop_gradient(f)

      if self._hparams.ll_l2_regularizer > 0.0:
        all_losses = [loss]

        for weight in weights:
          weight_loss = tf.nn.l2_loss(weight)
          weight_loss_sum = tf.reduce_sum(weight_loss)
          weight_loss_scaled = weight_loss_sum * self._hparams.cue_nn_l2_regularizer
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
        self._dual.get_pl('batch_type'): batch_type
    })

  def add_fetches(self, fetches, batch_type='training'):
    """Adds ops that will get evaluated."""
    names = ['loss', 'encoding', 'decoding', 'inputs']

    if batch_type == 'training':
      names.extend(['training'])

    self._dual.add_fetches(fetches, names)

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""

    # Loss (not a tensor)
    self._loss = fetched[self.name]['loss']

    names = ['encoding', 'decoding', 'inputs']
    self._dual.set_fetches(fetched, names)

    # Summaries
    super().set_fetches(fetched, batch_type)
