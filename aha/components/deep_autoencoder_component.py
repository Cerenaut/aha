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

"""DeepAutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils
from pagi.utils.dual import DualData
from pagi.utils.np_utils import np_write_filters
from pagi.utils.layer_utils import type_activation_fn
from pagi.components.summarize_levels import SummarizeLevels
from pagi.components.autoencoder_component import AutoencoderComponent

from aha.utils.generic_utils import build_kernel_initializer


class DeepAutoencoderComponent(AutoencoderComponent):
  """Deep Autoencoder with untied weights (and untied biases)."""

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        learning_rate=0.005,
        loss_type='mse',
        num_layers=3,
        nonlinearity=['relu', 'relu', 'relu'],
        output_nonlinearity='sigmoid',
        batch_size=64,
        filters=[128, 64, 32],

        pm_type='none',               # Not relevant; use pm_raw_type
        pm_raw_type='none',           # 'none' or 'nn': map stable PC patterns back to VC input (image space)
        pm_l1_size=100,               # hidden layer of PM path (pattern mapping)
        pm_raw_hidden_size=[100],
        pm_raw_l2_regularizer=0.0,
        pm_raw_nonlinearity='leaky_relu',
        pm_noise_type='s',            # 's' for salt, 'sp' for salt + pepper
        pm_train_with_noise=0.0,
        pm_train_with_noise_pp=0.0,
        pm_train_dropout_input_keep_prob=1.0,
        pm_train_dropout_hidden_keep_prob=[1.0],

        optimizer='adam',
        momentum=0.9,
        momentum_nesterov=False,
        summarize_level=SummarizeLevels.ALL.value,
        max_outputs=3   # Number of outputs in TensorBoard
    )

  def __init__(self):
    super().__init__()

    self._use_pm = False
    self._use_pm_raw = False

  @property
  def use_input_cue(self):
    return False

  @property
  def use_pm(self):
    return self._use_pm

  @property
  def use_pm_raw(self):
    return self._use_pm_raw

  def use_nn_in_pr_path(self):
    return True

  def get_ec_out_raw_op(self):
    return None

  def variables_networks(self, outer_scope):
    vars_nets = []

    # Selectively include/exclude optimizer parameters
    optim_ae = False
    optim_pm_raw = True

    vars_nets += self._variables_encoder(outer_scope)
    vars_nets += self._variables_decoder(outer_scope)

    if optim_ae:
      vars_nets += self._variables_ae_optimizer(outer_scope)

    if self.use_pm_raw:
      vars_nets += self._variables_pm_raw(outer_scope)
      if optim_pm_raw:
        vars_nets += self._variables_pm_raw_optimizer(outer_scope)

    return vars_nets

  def get_input(self, batch_type):
    """Unlike Hopfield, a standard component only has the one input, so return that regardless of batch type."""
    del batch_type
    return self.get_inputs()

  def get_losses_pm(self, default=0):
    loss = self._dual.get_values('pm_loss')
    loss_raw = self._dual.get_values('pm_loss_raw')

    if loss is None:
      loss = default

    if loss_raw is None:
      loss_raw = default

    return loss, loss_raw

  def build(self, input_values, input_shape, hparams, name='deep_ae', input_cue_raw=None, encoding_shape=None):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
    """
    self._name = name
    self._hidden_name = 'hidden'
    self._hparams = hparams
    self._dual = DualData(self._name)
    self._summary_training_op = None
    self._summary_encoding_op = None
    self._summary_values = None
    self._weights = None

    self._input_shape = input_shape
    self._input_values = input_values
    self._encoding_shape = encoding_shape

    if self._encoding_shape is None:
      self._encoding_shape = self._create_encoding_shape_4d(input_shape)  # .as_list()

    if self._hparams.pm_raw_type != 'none' and input_cue_raw is not None:
      self._use_pm_raw = True
      self._input_cue_raw = input_cue_raw

    self._batch_type = None

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      # 1) Build the deep autoencoder
      self._build()

      # 2) build Pattern Mapping
      # ---------------------------------------------------
      if self._use_pm or self._use_pm_raw:
        self._build_pm()

      self.reset()

  def _create_encoding_shape_4d(self, input_shape):  # pylint: disable=W0613
    """Put it into convolutional geometry: [batches, filter h, filter w, filters]"""
    return [self._hparams.batch_size, 1, 1, self._hparams.filters[-1]]

  def _build_optimizer(self, loss_op, training_op_name, scope=None):
    """Minimise loss using initialised a tf.train.Optimizer."""

    logging.info("-----------> Adding optimiser for op {0}".format(loss_op))

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

  def _build(self):
    """Build the autoencoder network"""

    self._batch_type = tf.placeholder_with_default(input='training', shape=[], name='batch_type')

    self._dual.set_op('inputs', self._input_values)

    # input_shape = self._input_values.get_shape().as_list()
    # output_shape = np.prod(input_shape[1:])

    # kernel_initializer = build_kernel_initializer('xavier')

    # assert self._hparams.num_layers == len(self._hparams.filters)
    # assert self._hparams.num_layers == len(self._hparams.nonlinearity)

    # with tf.variable_scope('encoder'):
    #   encoder_output = tf.layers.flatten(self._input_values)

    #   print('input', encoder_output)
    #   for i in range(self._hparams.num_layers):
    #     encoder_output = tf.layers.dense(encoder_output, self._hparams.filters[i],
    #                                      activation=type_activation_fn(self._hparams.nonlinearity[i]),
    #                                      kernel_initializer=kernel_initializer)
    #     print('encoder', i, encoder_output)

    #   self._dual.set_op('encoding', tf.stop_gradient(encoder_output))

    # with tf.variable_scope('decoder'):
    #   decoder_output = encoder_output
    #   decoder_filters = self._hparams.filters[:-1][::-1]  # Remove last filter (bottleneck), reverse filters
    #   decoder_filters += [output_shape]

    #   decoder_nonlinearity = self._hparams.nonlinearity[:-1][::-1]
    #   decoder_nonlinearity += [self._hparams.output_nonlinearity]

    #   for i in range(self._hparams.num_layers):
    #     decoder_output = tf.layers.dense(encoder_output, decoder_filters[i],
    #                                      activation=type_activation_fn(decoder_nonlinearity[i]),
    #                                      kernel_initializer=kernel_initializer)
    #     print('decoder', i, decoder_output)

    #   output = tf.reshape(decoder_output, [-1] + input_shape[1:])
    #   print('output', output)

    #   self._dual.set_op('decoding', output)
    #   self._dual.set_op('output', tf.stop_gradient(output))

    # # Build loss and optimizer
    # loss = self._build_loss_fn(self._input_values, output)
    # self._dual.set_op('loss', loss)
    # self._build_optimizer(loss, 'training', scope='ae')

  def _build_pm(self):
    """Preprocess the inputs and build the pattern mapping components."""

    def normalize(x):
      return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    # map to input
    # x_nn = self._dual.get_op('output')  # output of Deep AE
    x_nn = self._dual.get_op('inputs')
    x_nn = tf.layers.flatten(x_nn)

    x_nn = normalize(x_nn)

    # Apply noise during training, to regularise / test generalisation
    # --------------------------------------------------------------------------
    if self._hparams.pm_noise_type == 's':  # salt noise
      x_nn = tf.cond(
          tf.equal(self._batch_type, 'training'),
          lambda: image_utils.add_image_salt_noise_flat(x_nn, None,
                                                        noise_val=self._hparams.pm_train_with_noise,
                                                        noise_factor=self._hparams.pm_train_with_noise_pp,
                                                        mode='replace'),
          lambda: x_nn
      )

    elif self._hparams.pm_noise_type == 'sp':  # salt + pepper noise
      # Inspired by denoising AE.
      # Add salt+pepper noise to mimic missing/extra bits in PC space.
      # Use a fairly high rate of noising to mitigate few training iters.
      x_nn = tf.cond(
          tf.equal(self._batch_type, 'training'),
          lambda: image_utils.add_image_salt_pepper_noise_flat(x_nn, None,
                                                               salt_val=self._hparams.pm_train_with_noise,
                                                               pepper_val=-self._hparams.pm_train_with_noise,
                                                               noise_factor=self._hparams.pm_train_with_noise_pp),
          lambda: x_nn
      )

    else:
      raise NotImplementedError('PM noise type not supported: ' + str(self._hparams.noise_type))

    # apply dropout during training
    keep_prob = self._hparams.pm_train_dropout_input_keep_prob
    x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                   lambda: tf.nn.dropout(x_nn, keep_prob),
                   lambda: x_nn)

    # Build PM
    # --------------------------------------------------------------------------
    if self.use_pm:
      ec_in = self._input_cue
      output_nonlinearity = type_activation_fn('leaky_relu')
      ec_out = self._build_pm_core(x=x_nn, target=ec_in,
                                   hidden_size=self._hparams.pm_l1_size,
                                   non_linearity1=tf.nn.leaky_relu,
                                   non_linearity2=output_nonlinearity,
                                   loss_fn=tf.losses.mean_squared_error)
      self._dual.set_op('ec_out', ec_out)

    if self._use_pm_raw:
      ec_in = self._input_cue_raw
      output_nonlinearity = type_activation_fn(self._hparams.pm_raw_nonlinearity)
      ec_out_raw = self._build_pm_core(x=x_nn, target=ec_in,
                                       hidden_size=self._hparams.pm_raw_hidden_size,
                                       non_linearity1=tf.nn.leaky_relu,
                                       non_linearity2=output_nonlinearity,
                                       loss_fn=tf.losses.mean_squared_error,
                                       name_suffix="_raw")
      self._dual.set_op('ec_out_raw', ec_out_raw)


  def _build_pm_core(self, x, target, hidden_size, non_linearity1, non_linearity2, loss_fn, name_suffix=""):
    """Build the layers of the PM network, with optional L2 regularization."""
    target_shape = target.get_shape().as_list()
    target_size = np.prod(target_shape[1:])
    l2_size = target_size

    weights = []
    scope = 'pm' + name_suffix
    with tf.variable_scope(scope):
      out = x
      keep_prob = self._hparams.pm_train_dropout_hidden_keep_prob

      # Build encoding layers
      for i, num_units in enumerate(hidden_size):
        hidden_layer = tf.layers.Dense(units=num_units, activation=non_linearity1)
        out = hidden_layer(out)

        weights.append(hidden_layer.weights[0])
        weights.append(hidden_layer.weights[1])

        # apply dropout during training
        out = tf.cond(tf.equal(self._batch_type, 'training'),
                      lambda: tf.nn.dropout(out, keep_prob[i]),
                      lambda: out)

      # Store final hidden state
      self._dual.set_op('encoding', tf.stop_gradient(out))
      self._dual.set_op('decoding', tf.stop_gradient(out))

      # Build output layer
      f_layer = tf.layers.Dense(units=l2_size, activation=non_linearity2)
      f = f_layer(out)

      weights.append(f_layer.weights[0])
      weights.append(f_layer.weights[1])

      y = tf.stop_gradient(f)   # ensure gradients don't leak into other nn's in PC
      self._dual.set_op('output', y)

    target_flat = tf.reshape(target, shape=[-1, target_size])
    loss = loss_fn(f, target_flat)

    self._dual.set_op('loss', loss)
    self._dual.set_op('pm_loss' + name_suffix, loss)

    if self._hparams.pm_raw_l2_regularizer > 0.0:
      all_losses = [loss]

      for weight in weights:
        weight_loss = tf.nn.l2_loss(weight)
        weight_loss_sum = tf.reduce_sum(weight_loss)
        weight_loss_scaled = weight_loss_sum * self._hparams.pm_raw_l2_regularizer
        all_losses.append(weight_loss_scaled)

      all_losses_op = tf.add_n(all_losses, name='total_pm_loss')
      self._build_optimizer(all_losses_op, 'training_pm' + name_suffix, scope)
    else:
      self._build_optimizer(loss, 'training_pm' + name_suffix, scope)

    return y

  @staticmethod
  def _variables_ae_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer/ae")

  @staticmethod
  def _variables_pm_raw_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer/pm_raw")

  @staticmethod
  def _variables_encoder(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/encoder"
    )

  @staticmethod
  def _variables_decoder(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/decoder"
    )

  @staticmethod
  def _variables_pm_raw(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/pm_raw"
    )

  # OP ACCESS ------------------------------------------------------------------
  def get_encoding_op(self):
    return self._dual.get_op('encoding')

  def get_decoding_op(self):
    return self._dual.get_op('decoding')

  def get_ec_out_raw_op(self):
    return self._dual.get_op('ec_out_raw')

  # MODULAR INTERFACE ------------------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    if batch_type == 'training':
      self.update_training_dict(feed_dict)
    if batch_type == 'encoding':
      self.update_encoding_dict(feed_dict)

  def add_fetches(self, fetches, batch_type='training'):
    if batch_type == 'training':
      self.add_training_fetches(fetches)
    if batch_type == 'encoding':
      self.add_encoding_fetches(fetches)

  def set_fetches(self, fetched, batch_type='training'):
    if batch_type == 'training':
      self.set_training_fetches(fetched)
    if batch_type == 'encoding':
      self.set_encoding_fetches(fetched)

  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:
        if batch_type == 'training':
          self.build_training_summaries()
        if batch_type == 'encoding':
          self.build_encoding_summaries()

  def write_summaries(self, step, writer, batch_type='training'):
    """Write the summaries fetched into _summary_values"""
    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)
      writer.flush()

  # TRAINING ------------------------------------------------------------------
  def update_training_dict(self, feed_dict):
    feed_dict.update({
        self._batch_type: 'training'
    })

  def add_training_fetches(self, fetches):
    # names = ['loss', 'training', 'encoding', 'decoding', 'inputs']
    names = ['loss', 'encoding', 'decoding', 'inputs']

    if self._use_pm_raw:
      names.extend(['training_pm_raw', 'ec_out_raw'])

    self._dual.add_fetches(fetches, names)

    if self._summary_training_op is not None:
      fetches[self._name]['summaries'] = self._summary_training_op

  def set_training_fetches(self, fetched):
    self_fetched = fetched[self._name]
    self._loss = self_fetched['loss']

    names = ['encoding', 'decoding', 'inputs']

    if self._use_pm_raw:
      names.extend(['ec_out_raw'])

    self._dual.set_fetches(fetched, names)

    if self._summary_training_op is not None:
      self._summary_values = fetched[self._name]['summaries']

  # ENCODING ------------------------------------------------------------------
  def update_encoding_dict(self, feed_dict):
    feed_dict.update({
        self._batch_type: 'encoding'
    })

  def add_encoding_fetches(self, fetches):
    names = ['encoding', 'decoding', 'inputs']

    if self._use_pm_raw:
      names.extend(['pm_loss_raw', 'ec_out_raw'])

    self._dual.add_fetches(fetches, names)

    if self._summary_encoding_op is not None:
      fetches[self._name]['summaries'] = self._summary_encoding_op

  def set_encoding_fetches(self, fetched):
    names = ['encoding', 'decoding', 'inputs']

    if self._use_pm_raw:
      names.extend(['pm_loss_raw', 'ec_out_raw'])

    self._dual.set_fetches(fetched, names)

    if self._summary_encoding_op is not None:
      self._summary_values = fetched[self._name]['summaries']

  def get_inputs(self):
    return self._dual.get_values('inputs')

  def get_encoding(self):
    return self._dual.get_values('encoding')

  def get_decoding(self):
    return self._dual.get_values('decoding')

  def get_ec_out_raw(self):
    return self._dual.get_values('ec_out_raw')

  def get_batch_type(self):
    return self._batch_type

  # SUMMARIES ------------------------------------------------------------------
  def write_filters(self, session, folder=None):
    pass

  def build_training_summaries(self):
    with tf.name_scope('training'):
      summaries = self._build_summaries()
      if len(summaries) > 0:
        self._summary_training_op = tf.summary.merge(summaries)
      return self._summary_training_op

  def build_encoding_summaries(self):
    with tf.name_scope('encoding'):
      summaries = self._build_summaries()
      if len(summaries) > 0:
        self._summary_encoding_op = tf.summary.merge(summaries)
      return self._summary_encoding_op

  def _build_summarise_pm(self, summaries, max_outputs):
    if not (self.use_pm_raw or self._use_pm):
      return

    with tf.name_scope('pm'):

      # original vc for visuals
      if self.use_pm_raw:
        ec_in = self._input_cue_raw
        ec_out = self._dual.get_op('ec_out_raw')
        ec_recon = image_utils.concat_images([ec_in, ec_out], self._hparams.batch_size)
        summaries.append(tf.summary.image('ec_recon_raw', ec_recon, max_outputs=max_outputs))

        pm_loss_raw = self._dual.get_op('pm_loss_raw')
        summaries.append(tf.summary.scalar('pm_loss_raw', pm_loss_raw))

  def _build_summaries(self):
    """Build the summaries for TensorBoard."""

    # summarise_stuff = ['general', 'pm']
    summarise_stuff = ['pm']

    max_outputs = self._hparams.max_outputs
    summaries = []

    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    if 'general' in summarise_stuff:
      encoding_op = self.get_encoding_op()
      decoding_op = self.get_decoding_op()

      summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)

      input_summary_reshape = tf.reshape(self._input_values, summary_input_shape)
      decoding_summary_reshape = tf.reshape(decoding_op, summary_input_shape)

      summary_reconstruction = tf.concat([input_summary_reshape, decoding_summary_reshape], axis=1)
      reconstruction_summary_op = tf.summary.image('reconstruction', summary_reconstruction,
                                                   max_outputs=max_outputs)
      summaries.append(reconstruction_summary_op)

      # show input on it's own
      input_alone = True
      if input_alone:
        summaries.append(tf.summary.image('input', input_summary_reshape, max_outputs=max_outputs))

      summaries.append(self._summary_hidden(encoding_op, 'encoding', max_outputs))

      # Loss
      loss_summary = tf.summary.scalar('loss', self._dual.get_op('loss'))
      summaries.append(loss_summary)

    if 'pm' in summarise_stuff:
      self._build_summarise_pm(summaries, max_outputs)

    return summaries

  def _summary_hidden(self, hidden, name, max_outputs=3):
    """Return a summary op of a 'square as possible' image of hidden, the tensor for the hidden state"""
    hidden_shape_4d = self._hidden_image_summary_shape()  # [batches, height=1, width=filters, 1]
    summary_reshape = tf.reshape(hidden, hidden_shape_4d)
    summary_op = tf.summary.image(name, summary_reshape, max_outputs=max_outputs)
    return summary_op
