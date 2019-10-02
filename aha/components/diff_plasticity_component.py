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

"""DifferentiablePlasticityComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.utils import tf_utils, image_utils
from pagi.utils.image_utils import add_square_as_square
from pagi.utils.layer_utils import activation_fn
from pagi.utils.dual import DualData
from pagi.utils.np_utils import np_write_filters
from pagi.utils.tf_utils import tf_print
from pagi.utils.tf_utils import tf_build_stats_summaries_short

from pagi.components.summary_component import SummaryComponent


MUTE_DEBUG_GRAPH = True

class DifferentiablePlasticityComponent(SummaryComponent):
  """
  Differentiable Plasticity algorithm from Uber, Miconi

  WARNINGS:
    - currently number of neurons (filters) must be equal to input vector size.
    - currently only returns correct dimensions if input shape = [batch size, input sample area]

  encoding == no equivalent here
  decoding == the output of the layer

  batch_types:
    to be consistent with ae, using 'encoding' in place of 'testing'
    So:
      training  - use BPIT to update weights
      encoding  - just use for inference

  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters (use values closest to Miconi as default)."""
    return tf.contrib.training.HParams(
        batch_size=5,
        learning_rate=0.0003,
        loss_type='sse',             # sum square error (only option is sse)
        nonlinearity='tanh',         # tanh or sigmoid
        filters=1024,
        bias_neurons=0,              # add this many 'active' bias neurons
        bias=False,                  # include a bias value (to be trained)
        use_batch_transformer=True,  #
        bt_presentation_repeat=2,    # number of times the total sequence of repeats with blanks, is repeated
        bt_sample_repeat=6,          # number of repeats of each original sample (1 = identical to input)
        bt_blank_repeat=4,           # number of zero samples between each original sample
        bt_amplify_factor=20,        # amplify input by this amount
        bt_degrade=True,             # randomly select a sample from batch, degrade and append it & non-degraded sample
        bt_degrade_repeat=6,
        bt_degrade_value=0.0,        # when degrading, set pixel to this value
        bt_degrade_factor=0.5,       # what proportion of bits to knockout
        bt_degrade_type='random',    # options: 'random' = randomly degrade,
                                     # 'vertical' = degrade a random half along vertical symmetry,
                                     # 'horizontal' = same but horizontal symmetry
        input_sparsity=0.5,
        max_outputs=3
    )

  def __init__(self):
    self._name = None
    self._hparams = None
    self._dual = None
    self._summary_op = None
    self._summary_values = None
    self._get_weights_op = None
    self._input_shape_visualisation = None
    self._input_values = None
    self._loss = None
    self._active_bits = None
    self._blank_indices = []

  def build(self, input_values, input_shape_visualisaion, hparams, name='diff_plasticity'):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape_visualisaion: The shape of the input, for display (internal is vectorized)
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
    """
    self._name = name
    self._hparams = hparams
    self._dual = DualData(self._name)
    self._summary_op = None
    self._summary_values = None
    self._get_weights_op = None
    self._input_shape_visualisation = input_shape_visualisaion
    self._input_values = input_values

    length_sparse = int(self._hparams.filters * self._hparams.input_sparsity)
    self._active_bits = int(self._hparams.filters - length_sparse)  # do it this way to match the int rounding in generator

    self._build()

  @property
  def name(self):
    return self._name

  def get_loss(self):
    return self._loss

  def get_dual(self):
    return self._dual

  def _build(self):
    """Build the component"""

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      input_tensor = self._input_values

      if self._hparams.use_batch_transformer:
        input_tensor = self._build_batch_transformer(self._input_values)

      # add bias neurons - important that it is after batch_transformer, and therefore the degradations
      num_bias_neurons = self._hparams.bias_neurons
      if num_bias_neurons > 0:
        num_batches = input_tensor.shape.as_list()[0]
        input_shape_with_bias = [num_batches, num_bias_neurons]
        bias_neurons = tf.ones(shape=input_shape_with_bias, dtype=tf.float32)
        input_tensor = tf.concat([input_tensor, bias_neurons], 1)

        # add bias to the raw input also, so that visualisations align (in terms of expectations and sizes)
        num_batches = self._input_values.shape.as_list()[0]
        input_shape_with_bias = [num_batches, num_bias_neurons]
        bias_neurons = tf.ones(shape=input_shape_with_bias, dtype=tf.float32)
        self._input_values = tf.concat([self._input_values, bias_neurons], 1)

      _, testing = self._build_rnn(input_tensor)

      # output fork - ths path doesn't accumulate gradients
      # -----------------------------------------------------------------
      stop_gradient = tf.stop_gradient(testing)
      self._dual.set_op('output', stop_gradient)

  def _build_batch_transformer(self, input_tensor):
    """
    input_tensor = input batch, shape = [input batch size, sample tensor shape]
    The output is transformed, shape = [output batch size, sample tensor shape]
    Adds a stop gradient at the end because nothing in here is trainable

    - control number of:
      - repeats of each sample
      - repeats of blanks between groups of repeated samples
      - number of presentations of samples (with repeats)
    - randomly select and degraded a sample, append at the end, together with the un-corrupted copy

    Shuffle batch order for each presentation.
    """

    self._dual.set_op('bt_input', input_tensor)

    # get hyper params
    sample_repeat = self._hparams.bt_sample_repeat
    blank_repeat = self._hparams.bt_blank_repeat
    presentation_repeat = self._hparams.bt_presentation_repeat
    is_degrade = self._hparams.bt_degrade
    degrade_repeat = self._hparams.bt_degrade_repeat
    degrade_type = self._hparams.bt_degrade_type
    degrade_value = self._hparams.bt_degrade_value
    degrade_factor = self._hparams.bt_degrade_factor

    # note: x_batch = input batch, y_batch = transformed batch

    with tf.variable_scope('batch_transformer'):

      # convert input to shape [batch, samples]
      input_shape = input_tensor.get_shape().as_list()
      input_area = np.prod(input_shape[1:])
      batch_input_shape = (-1, input_area)

      input_vector = tf.reshape(input_tensor, batch_input_shape, name='input_vector')
      logging.debug(input_vector)

      x_batch_length = input_shape[0]
      y_batch_length = presentation_repeat * x_batch_length * (sample_repeat + blank_repeat) + \
                       (1 + degrade_repeat if is_degrade else 0)

      self._blank_indices = []
      for p in range(presentation_repeat):
        start_pres = p * x_batch_length * (sample_repeat + blank_repeat)
        for s in range(x_batch_length):
          blank_start = start_pres + s*(sample_repeat+blank_repeat) + sample_repeat
          self._blank_indices.append([blank_start, blank_start + blank_repeat-1])

      # start with all blanks, in this case zero tensors
      y_batch = tf.get_variable(initializer=tf.zeros(shape=[y_batch_length, input_area]),
                                trainable=False,
                                name='blanks')

      # use scatter updates to fill with repeats, can not do 1-to-many, so need to do it
      # `pres_repeat * sample_repeat` times
      presentation_length = x_batch_length * (sample_repeat + blank_repeat)
      for p in range(presentation_repeat):

        input_vector = tf.random_shuffle(input_vector)

        for i in range(sample_repeat):
          x2y = []  # the idx itself is the x_idx, val = y_idx
          for x_idx in range(x_batch_length):
            y_idx = (p * presentation_length) + x_idx * (sample_repeat + blank_repeat) + i
            x2y.append(y_idx)
          xy_scatter_map = tf.constant(value=x2y, name='x_y_scatter_' + str(i))
          y_batch = tf.scatter_update(y_batch, xy_scatter_map, input_vector, name="sample_repeat")

      # append degraded and non-degraded samples
      if is_degrade:
        # randomly choose one of the input vectors
        input_shuffled = tf.random_shuffle(input_vector)
        target = input_shuffled[0]

        if degrade_type == 'horizontal':
          degraded = image_utils.degrade_image(input_shuffled, label=None, degrade_type='horizontal',
                                               degrade_value=degrade_value)[0]
        elif degrade_type == 'vertical':
          raise NotImplementedError('vertical degradation not implemented')
        elif degrade_type == 'random':

          # This next commented out line, caused major malfunction (result was passed to degrade in place of the whole batch - but i have no idea why)
          # degraded_samples = tf.reshape(target, batch_input_shape)  # for efficiency, only degrade one image

          min_value_0 = True
          if min_value_0 is not True:
            degraded = image_utils.degrade_image(image=input_shuffled, label=None, degrade_type='random',
                                                 degrade_value=degrade_value,
                                                 degrade_factor=degrade_factor)[0]
          else:
            # degrade the high bits (not the bits that are already zero)
            eps = 0.01
            degrade_mask = tf.greater(target, 1.0 - eps)
            degrade_mask = tf.to_float(degrade_mask)
            degrade_mask = tf_print(degrade_mask, "degrade_mask", mute=True)

            degraded = tf_utils.degrade_by_mask(input_tensor=input_shuffled,
                                                num_active=self._active_bits,
                                                degrade_mask=degrade_mask,
                                                degrade_factor=degrade_factor,
                                                degrade_value=degrade_value)[0]

        else:
          raise NotImplementedError('Unknown degradation type.')

        degraded_repeated = tf.ones([degrade_repeat, input_area])
        degraded_repeated = degraded_repeated * degraded

        target = tf.reshape(target, [1, input_area])
        degraded_and_target = tf.concat([degraded_repeated, target], 0)

        index_map = []
        for i in range(degrade_repeat+1):
          index_map.insert(0, y_batch_length - 1 - i)

        y_batch = tf.scatter_update(y_batch, tf.constant(index_map), degraded_and_target, name="degradetarget")
        y_batch = tf.stop_gradient(y_batch)

    self._dual.set_op('bt_output', y_batch)
    return y_batch

  def _build_rnn(self, input_tensor):
    """
    Build the encoder network

    input_tensor = 1 batch = 1 episode (batch size, #neurons)
    Assumes second last item is degraded input, last is target
    """

    w_trainable = False
    x_shift_trainable = False
    eta_trainable = True

    input_shape = input_tensor.get_shape().as_list()
    input_area = np.prod(input_shape[1:])
    batch_input_shape = (-1, input_area)

    filters = self._hparams.filters + self._hparams.bias_neurons
    hidden_size = [filters]
    weights_shape = [filters, filters]

    with tf.variable_scope("rnn"):
      init_state_pl = self._dual.add('init_pl', shape=hidden_size, default_value=0).add_pl()
      init_hebb_pl = self._dual.add('hebb_init_pl', shape=weights_shape, default_value=0).add_pl()

      # ensure init placeholders are being reset every iteration
      init_hebb_pl = tf_print(init_hebb_pl, "Init Hebb:", summarize=100, mute=True)

      # Input reshape: Ensure flat (vector) x batch size input (batches, inputs)
      # -----------------------------------------------------------------
      input_vector = tf.reshape(input_tensor, batch_input_shape, name='input_vector')

      # unroll input into a series so that we can iterate over it easily
      x_series = tf.unstack(input_vector, axis=0, name="ep-series")  # batch_size of hidden_size

      # get the target and degraded samples
      target = input_vector[-1]
      target = tf_print(target, "TARGET\n", mute=True)
      degraded_extracted = input_vector[-2]
      degraded_extracted = tf_print(degraded_extracted, "DEGRADED-extracted\n", mute=True)
      self._dual.set_op('target', target)
      self._dual.set_op('degraded_raw', degraded_extracted)

      y_current = tf.reshape(init_state_pl, [1, filters], name="init-curr-state")
      hebb = init_hebb_pl

      with tf.variable_scope("slow-weights"):
        w_default = 0.01
        alpha_default = 0.1
        eta_default = 0.1
        x_shift_default = 0.01
        bias_default = 1.0 * w_default  # To emulate the Miconi method of having an additional input at 20 i.e.
        # it creates an output of 1.0, and this is multiplied by the weight (here we have straight bias, no weight)

        if w_trainable:
          w = tf.get_variable(name="w", initializer=(w_default * tf.random_uniform(weights_shape)))
        else:
          w = tf.zeros(weights_shape)

        alpha = tf.get_variable(name="alpha", initializer=(alpha_default * tf.random_uniform(weights_shape)))

        if eta_trainable:
          eta = tf.get_variable(name="eta", initializer=(eta_default * tf.ones(shape=[1])))
        else:
          eta = eta_default * tf.ones([1])

        if x_shift_trainable:
          x_shift = tf.get_variable(name="x_shift", initializer=(x_shift_default * tf.ones(shape=[1])))
        else:
          x_shift = 0

        self._dual.set_op('w', w)
        self._dual.set_op('alpha', alpha)
        self._dual.set_op('eta', eta)
        self._dual.set_op('x_shift', x_shift)

        if self._hparams.bias:
          bias = tf.get_variable(name="bias", initializer=(bias_default * tf.ones(filters)))
          self._dual.set_op('bias', bias)
          bias = tf_print(bias, "*** bias ***", mute=MUTE_DEBUG_GRAPH)

      with tf.variable_scope("layers"):
        hebb = tf_print(hebb, "*** initial hebb ***", mute=MUTE_DEBUG_GRAPH)
        y_current = tf_print(y_current, "*** initial state ***")
        w = tf_print(w, "*** w ***", mute=MUTE_DEBUG_GRAPH)
        alpha = tf_print(alpha, "*** alpha ***", mute=MUTE_DEBUG_GRAPH)

        i = 0
        last_x = None
        outer_first = None
        outer_last = None
        for x in x_series:
          # last sample is target, so don't process it again
          if i == len(x_series) - 1:    # [0:x, 1:d, 2:t], l=3
            break
          layer_name = "layer-" + str(i)
          with tf.variable_scope(layer_name):
            x = self._hparams.bt_amplify_factor * x
            x = tf_print(x, str(i) + ": x_input", mute=MUTE_DEBUG_GRAPH)
            y_current = tf_print(y_current, str(i) + ": y(t-1)", mute=MUTE_DEBUG_GRAPH)

            # neurons latch on as they have bidirectional connections
            # attempt to remove this issue by knocking out lateral connections
            remove = 'random'
            if remove == 'circular':
              diagonal_mask = tf.convert_to_tensor(np.tril(np.ones(weights_shape, dtype=np.float32), 0))
              alpha = tf.multiply(alpha, diagonal_mask)
            elif remove == 'random':
              size = np.prod(weights_shape[:])
              knockout_mask = np.ones(size)
              knockout_mask[:int(size / 2)] = 0
              np.random.shuffle(knockout_mask)
              knockout_mask = np.reshape(knockout_mask, weights_shape)
              alpha = tf.multiply(alpha, knockout_mask)

            # ---------- Calculate next output of the RNN
            weighted_sum = tf.add(tf.matmul(y_current - x_shift,
                                            tf.add(w, tf.multiply(alpha, hebb, name='lyr-mul'), name="lyr-add_w_ah"),
                                            name='lyr-mul-add-matmul'),
                                  x, "weighted_sum")

            if self._hparams.bias:
              weighted_sum = tf.add(weighted_sum, bias)  # weighted sum with bias

            y_next, _ = activation_fn(weighted_sum, self._hparams.nonlinearity)

            with tf.variable_scope("fast_weights"):
              # ---------- Update Hebbian fast weights
              # outer product of (yin * yout) = (current_state * next_state)
              outer = tf.matmul(tf.reshape(y_current, shape=[filters, 1]),
                                tf.reshape(y_next, shape=[1, filters]),
                                name="outer-product")
              outer = tf_print(outer, str(i) + ": *** outer = y(t-1) * y(t) ***", mute=MUTE_DEBUG_GRAPH)

              if i == 1:  # first outer is zero
                outer_first = outer
              outer_last = outer

              hebb = (1.0 - eta) * hebb + eta * outer
              hebb = tf_print(hebb, str(i) + ": *** hebb ***", mute=MUTE_DEBUG_GRAPH)

            # record for visualisation the output when presented with the last blank
            idx_blank_first = self._blank_indices[-1][0]
            idx_blank_last = self._blank_indices[-1][1]

            if i == idx_blank_first:
              blank_output_first = y_next
              self._dual.set_op('blank_output_first', blank_output_first)

            if i == idx_blank_last:
              blank_output_last = y_next
              self._dual.set_op('blank_output_last', blank_output_last)

            y_current = y_next
            last_x = x
            i = i + 1

      self._dual.set_op('hebb', hebb)
      self._dual.set_op('outer_first', outer_first)
      self._dual.set_op('outer_last', outer_last)

      last_x = tf_print(last_x, str(i) + ": LAST-X", mute=True)
      self._dual.set_op('degraded', last_x)

      output_pre_masked = tf.squeeze(y_current)
      self._dual.set_op('output_pre_masked', output_pre_masked)   # pre-masked output

    # External masking
    # -----------------------------------------------------------------
    with tf.variable_scope("masking"):
      mask_pl = self._dual.add('mask', shape=hidden_size, default_value=1.0).add_pl()
      y_masked = tf.multiply(y_current, mask_pl, name='y_masked')

    # Setup the training operations
    # -----------------------------------------------------------------
    with tf.variable_scope("optimizer"):
      loss_op = self._build_loss_op(y_masked, target)
      self._dual.set_op('loss', loss_op)

      self._optimizer = tf.train.AdamOptimizer(self._hparams.learning_rate)
      training_op = self._optimizer.minimize(loss_op,
                                             global_step=tf.train.get_or_create_global_step(), name='training_op')
      self._dual.set_op('training', training_op)

    return y_masked, y_masked

  def _build_loss_op(self, output, target):
    if self._hparams.loss_type == 'sse':
      losses = tf.subtract(output, target, name="losses")
      self._dual.set_op("losses", losses)
      return tf.reduce_sum(tf.square(losses))
    else:
      raise NotImplementedError(
          'Loss function not implemented: ' + str(self._hparams.loss_type))

  # OP ACCESS ------------------------------------------------------------------
  def get_decoding_op(self):
    """The equivalent of decoding is simply the output"""
    return self._dual.get_op('output')

  def get_encoding_op(self):
    raise NotImplementedError('The is no equivalent to encoding for this component')

  def get_loss_op(self):
    return self._dual.get_op('loss')

  # VALUE ACCESS ------------------------------------------------------------------
  def get_decoding(self):
    return self._dual.get_values('output')

  def get_encoding(self):
    raise NotImplementedError('The is no equivalent to encoding for this component')

  # MODULAR INTERFACE ------------------------------------------------------------------
  def reset(self):
    self._loss = 0.0
    logging.warning('reset() not yet properly implemented for this component.')

  def update_feed_dict(self, feed_dict, batch_type='training'):
    if batch_type == 'training':
      self.update_training_dict(feed_dict)
    if batch_type == 'encoding':
      self.update_testing_dict(feed_dict)

  def add_fetches(self, fetches, batch_type='training'):
    if batch_type == 'training':
      self.add_training_fetches(fetches)
    if batch_type == 'encoding':
      self.add_testing_fetches(fetches)

  def set_fetches(self, fetched, batch_type='training'):
    if batch_type == 'training':
      self.set_training_fetches(fetched)
    if batch_type == 'encoding':
      self.set_testing_fetches(fetched)

  def get_features(self, batch_type='training'):
    return self._dual.get_values('encoding')

  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:
        self._build_batchtype_summaries(batch_type=batch_type)  # same for all batch_types.... for now

  def write_summaries(self, step, writer, batch_type='training'):

    if batch_type == 'training':
      # debugging batch_transformer
      if self._hparams.use_batch_transformer:
        bt_input = self._dual.get_values('bt_input')
        bt_output = self._dual.get_values('bt_output')
        logging.debug("BT Input", bt_input[:4][:2])
        logging.debug("BT Output", bt_output[:4][:2])

    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)
      writer.flush()

  def _update_with_dual(self, feed_dict, name):
    """Convenience function to update a feed dict with a dual, using its placeholder and values."""
    dual = self._dual.get(name)
    dual_pl = dual.get_pl()
    dual_values = dual.get_values()
    feed_dict.update({
        dual_pl: dual_values,
    })

  # TRAINING ------------------------------------------------------------------
  def update_training_dict(self, feed_dict):
    """Update the feed_dict for training mode (batch_types)"""
    self._update_with_dual(feed_dict, 'hebb_init_pl')  # re-initialize hebbian at the start of episode
    self._update_with_dual(feed_dict, 'init_pl')       # re-initialize rnn state at the start of episode
    self._update_with_dual(feed_dict, 'mask')          # mask

  def add_training_fetches(self, fetches):
    """Add the fetches (ops to be evaluated) for training mode (batch_types)"""
    fetches[self._name] = {
        'loss': self._dual.get_op('loss'),          # the calculation of loss
        'training': self._dual.get_op('training'),  # the optimisation
        'output': self._dual.get_op('output'),      # the output value
        # debugging
        'target': self._dual.get_op('target'),
        'degraded': self._dual.get_op('degraded')
    }

    if self._hparams.use_batch_transformer:
      fetches[self._name]['bt_input'] = self._dual.get_op('bt_input')
      fetches[self._name]['bt_output'] = self._dual.get_op('bt_output')

    if self._summary_op is not None:
      fetches[self._name]['summaries'] = self._summary_op

  def set_training_fetches(self, fetched):
    """Add the fetches for training - the ops whose values will be available"""
    self_fetched = fetched[self._name]
    self._loss = self_fetched['loss']

    names = ['loss', 'output', 'target', 'degraded']

    if self._hparams.use_batch_transformer:
      names = names + ['bt_input', 'bt_output']

    self._dual.set_fetches(fetched, names)

    if self._summary_op is not None:
      self._summary_values = fetched[self._name]['summaries']

  # TESTING ------------------------------------------------------------------
  def update_testing_dict(self, feed_dict):
    self.update_training_dict(feed_dict)  # do the same for testing and training

  def add_testing_fetches(self, fetches):
    fetches[self._name] = {
        'loss': self._dual.get_op('loss'),  # the calculation of loss
        'output': self._dual.get_op('output')  # the output value
    }

    if self._summary_op is not None:
      fetches[self._name]['summaries'] = self._summary_op

  def set_testing_fetches(self, fetched):
    self_fetched = fetched[self._name]
    self._loss = self_fetched['loss']

    names = ['loss', 'output']
    self._dual.set_fetches(fetched, names)

    if self._summary_op is not None:
      self._summary_values = fetched[self._name]['summaries']

  # SUMMARIES ------------------------------------------------------------------
  def write_filters(self, session):
    """Write the learned filters to disk."""

    w = self._dual.get_op('w')
    weights_values = session.run(w)
    weights_transpose = np.transpose(weights_values)

    filter_height = self._input_shape_visualisation[1]
    filter_width = self._input_shape_visualisation[2]
    np_write_filters(weights_transpose, [filter_height, filter_width])

  def _build_batchtype_summaries(self, batch_type):
    """Same for all batch types right now. Other components have separate method for each batch_type."""
    with tf.name_scope(batch_type):
      summaries = self._build_summaries()
      self._summary_op = tf.summary.merge(summaries)
      return self._summary_op

  def _build_summaries(self):
    """
    Build the summaries for TensorBoard.
    Note that the outer scope has already been set by `_build_summaries()`.
      i.e. 'component_name' / 'summary name'
    """
    max_outputs = 3
    summaries = []

    # images
    # ------------------------------------------------
    summary_input_shape = image_utils.get_image_summary_shape(self._input_shape_visualisation)

    # input images
    input_summary_reshape = tf.reshape(self._input_values, summary_input_shape, name='input_summary_reshape')
    input_summary_op = tf.summary.image('input_images', input_summary_reshape, max_outputs=max_outputs)
    summaries.append(input_summary_op)

    # degraded, target and completed images, and histograms where relevant
    target = self._dual.get_op('target')
    degraded = self._dual.get_op('degraded')
    decoding_op = self.get_decoding_op()

    output_hist = tf.summary.histogram("output", decoding_op)
    summaries.append(output_hist)

    input_hist = tf.summary.histogram("input", self._input_values)
    summaries.append(input_hist)

    # network output when presented with blank
    blank_output_first = self._dual.get_op('blank_output_first')
    blank_first = tf.summary.image('blank_first', tf.reshape(blank_output_first, summary_input_shape))
    summaries.append(blank_first)

    blank_output_last = self._dual.get_op('blank_output_last')
    blank_last = tf.summary.image('blank_last', tf.reshape(blank_output_last, summary_input_shape))
    summaries.append(blank_last)
    
    with tf.name_scope('optimize'):
      completed_summary_reshape = tf.reshape(decoding_op, summary_input_shape, 'completed_summary_reshape')
      summaries.append(tf.summary.image('b_completed', completed_summary_reshape))

      if self._hparams.bt_degrade:
        degraded_summary_reshape = tf.reshape(degraded, summary_input_shape, 'degraded_summary_reshape')
        summaries.append(tf.summary.image('a_degraded', degraded_summary_reshape))

        target_summary_reshape = tf.reshape(target, summary_input_shape, 'target_summary_reshape')
        summaries.append(tf.summary.image('c_target', target_summary_reshape))

    # display slow weights as images and distributions
    with tf.name_scope('slow-weights'):
      w = self._dual.get_op('w')
      add_square_as_square(summaries, w, 'w')

      w_hist = tf.summary.histogram("w", w)
      summaries.append(w_hist)

      alpha = self._dual.get_op('alpha')
      add_square_as_square(summaries, alpha, 'alpha')

      alpha_hist = tf.summary.histogram("alpha", alpha)
      summaries.append(alpha_hist)

      if self._hparams.bias:
        bias = self._dual.get_op('bias')
        bias_image_shape, _ = image_utils.square_image_shape_from_1d(self._hparams.filters)
        bias_image = tf.reshape(bias, bias_image_shape, name='bias_summary_reshape')
        summaries.append(tf.summary.image('bias', bias_image))

        bias_hist = tf.summary.histogram("bias", bias)
        summaries.append(bias_hist)

      # eta
      eta_op = self._dual.get_op('eta')
      eta_scalar = tf.reduce_sum(eta_op)
      eta_summary = tf.summary.scalar('eta', eta_scalar)
      summaries.append(eta_summary)

      # x_shift
      x_shift_op = self._dual.get_op('x_shift')
      xs_scalar = tf.reduce_sum(x_shift_op)
      xs_summary = tf.summary.scalar('x_shift', xs_scalar)
      summaries.append(xs_summary)

    # display fast weights (eta and hebbian), as image, scalars and histogram
    with tf.name_scope('fast-weights'):

      # as images
      hebb = self._dual.get_op('hebb')
      add_square_as_square(summaries, hebb, 'hebb')

      # as scalars
      hebb_summary = tf_build_stats_summaries_short(hebb, 'hebb')
      summaries.append(hebb_summary)

      # as histograms
      hebb_hist = tf.summary.histogram("hebb", hebb)
      summaries.append(hebb_hist)

      hebb_per_neuron = tf.reduce_sum(tf.abs(hebb), 0)
      hebb_per_neuron = tf.summary.histogram('hebb_pn', hebb_per_neuron)
      summaries.append(hebb_per_neuron)

      # outer products
      outer_first = self._dual.get_op('outer_first')
      outer_last = self._dual.get_op('outer_last')
      add_square_as_square(summaries, outer_first, 'outer_first')
      add_square_as_square(summaries, outer_last, 'outer_last')

    # optimization related quantities
    with tf.name_scope('optimize'):
      # loss
      loss_op = self.get_loss_op()
      loss_summary = tf.summary.scalar('loss', loss_op)
      summaries.append(loss_summary)

      # losses as an image
      losses = self._dual.get_op("losses")
      shape = losses.get_shape().as_list()
      volume = np.prod(shape[1:])
      losses_image_shape, _ = image_utils.square_image_shape_from_1d(volume)
      losses_image = tf.reshape(losses, losses_image_shape)
      summaries.append(tf.summary.image('losses', losses_image))

    input_stats_summary = tf_build_stats_summaries_short(self._input_values, 'input-stats')
    summaries.append(input_stats_summary)

    return summaries
