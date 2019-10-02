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

"""
RecursiveSAEComponent class.


Terminology:

ws = Wx + b
Weighted sum.

z = mask(sigma(Wx + b)) = mask(sigma(ws))
The hidden state of the autoencoder.

If no activation function (sigma) is used, then z == ws == weighted sum

'encoding' = sparse z, i.e. after the filtering step for the sparse autoencoder (it is an additional non linearity)
This terminology is inherited from the autoencoder base class.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from pagi.utils import image_utils
from pagi.utils.tf_utils import tf_print
from pagi.utils.image_utils import add_square_as_square
from pagi.utils.layer_utils import activation_fn

from pagi.components.sparse_autoencoder_component import SparseAutoencoderComponent

TF_DBUG_MUTE = True


class RecursiveSAEComponent(SparseAutoencoderComponent):
  """
  Recursive Sparse Autoencoder.
  Inherits functionality of Sparse Autoencoder. Additionally, feeds the hidden layer values in as input
  (referred to as feedback input), together with the external input.
  """

  def __init__(self):
    super().__init__()
    self._input_mask = None   # this is 1d, and must be reshaped to apply to samples
    self._summary_recursive_values = None
    self._summary_recursive_op = None
    self._hidden_feedback_name = 'feedback_hidden'

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
      learning_rate=0.0001,
      loss_type='mse',
      nonlinearity='none',
      batch_size=250,
      filters=1024,
      optimizer='adam',
      momentum=0.9,
      momentum_nesterov=False,
      use_bias=False,

      # Additional sparse parameters:
      sparsity=25,
      sparsity_output_factor=1.0,

      # Additional recursive sae parameters
      use_feedback=True,
      normalise_feedback=True,
      sparse_feedback=False,  # provide sparse feedback or dense (before sparsification)
      optimize='fb',          # fb, ff, both:  ff=optimise only ff, fb=only fb, both=both
      reconstruct_fb_decoded=False,  # if optimize_fb, optimize recon of fb input, or mapping of input to itself(sparse)
      summary_long=False
    )

  def reset(self):
    super().reset()

    # set hidden state Dual 'values' (not the tensor op) to zero - this will be used for feedback
    self._dual.get('encoding').set_values_to(0.0)
    self._dual.get('z').set_values_to(0.0)

  ############################################
  # Override set/add fetches to ensure that 'z' is also filled in the off graph structure,
  # to be available for feeding back into Graph.

  def add_training_fetches(self, fetches):
    super().add_training_fetches(fetches)
    fetches[self._name]['z'] = self._dual.get_op('z')
    fetches[self._name]['fb_weights_adjust_op'] = self._dual.get_op('fb_weights_adjust_op')

  def add_encoding_fetches(self, fetches):
    super().add_encoding_fetches(fetches)
    fetches[self._name]['z'] = self._dual.get_op('z')

  def set_training_fetches(self, fetched):
    """Ensure that z is also filled in the off graph structure, to be available for feeding back into Graph."""
    super().set_training_fetches(fetched)

    names = ['z']
    self._dual.set_fetches(fetched, names)

  def set_encoding_fetches(self, fetched):
    """Ensure that z is also filled in the off graph structure, to be available for feeding back into Graph."""
    super().set_encoding_fetches(fetched)

    names = ['z']
    self._dual.set_fetches(fetched, names)

  ############################################

  def update_feed_dict(self, feed_dict, batch_type='training'):
    super().update_feed_dict(feed_dict, batch_type)

    # recursion: take hidden layer, and set as feedback
    if self._hparams.use_feedback:

      hidden_sparse = self._dual.get('encoding').get_values()
      hidden_dense = self._dual.get('z').get_values()

      if self._hparams.sparse_feedback:
        hidden_values = hidden_sparse
      else:
        hidden_values = hidden_dense

      feedback_dense_pl = self._dual.get('feedback_dense').get_pl()
      feedback_pl = self._dual.get('feedback').get_pl()
      feed_dict.update({
        feedback_pl: hidden_values,         # can be sparse or dense, depending on hparam
        feedback_dense_pl: hidden_dense     # always a sparse copy
      })

  def update_feed_dict_input_gain_pl(self, feed_dict, gain):
    input_gain_pl = self._dual.get('input_gain').get_pl()
    feed_dict.update({
      input_gain_pl: [gain]
    })

  def set_input_partial_pl(self, feed_dict):
    """Set the partial mask placeholder"""
    input_mask_pl = self._dual.get('input_mask').get_pl()
    feed_dict.update({
      input_mask_pl: self._input_mask
    })

  def reset_partial_mask(self, degree):
    """
    A mask is used to degrade the input to be a partial pattern. This creates a new random mask
    This is used for 'off graph' computations.
    Degree = fraction that is zero-ed out.
    """

    input_shape = self._input_values.shape.as_list()
    sample_size = np.prod(input_shape[1:])

    if self._input_mask is None:
      self._input_mask = np.ones(sample_size)
      self._input_mask[:int(degree * sample_size)] = 0
    np.random.shuffle(self._input_mask)

  def _add_feedback(self):
    # Input from previous timestep
    feedback_shape = [self._hparams.batch_size, self._hparams.filters]
    feedback_pl = self._dual.add('feedback', shape=feedback_shape, default_value=0.0).add_pl(default=True)

    feedback_dense_pl = self._dual.add('feedback_dense', shape=feedback_shape, default_value=0.0).add_pl(default=True)

    # Reduce the feedback input to a finite sum to prevent runaway increases in output
    if self._hparams.normalise_feedback:
      eps = 0.000001
      sum_feedback_values = tf.reduce_sum(feedback_pl, axis=[1], keep_dims=True) + eps
      unit_feedback_values = tf.divide(feedback_pl, sum_feedback_values)
      feedback_input = unit_feedback_values
    else:
      feedback_input = feedback_pl

# TODO WARNING These should already be added because of dual_add above!
    self._dual.set_op('feedback_dense', feedback_dense_pl)  # feedback 'dense' version ('z' from previous step)
    self._dual.set_op('feedback', feedback_pl)  # fb from prev step, dense ('z') or sparse ('encoding') on hparams

    self._dual.set_op('feedback_input', feedback_input)           # feedback after normalization

    return feedback_input

  def _build_encoding(self, input_tensor, mask_pl=None):
    """
    Build combined encoding of feedforward and feedback inputs.
    'add_bias' is honored for the feedforward, but feedback is set to NO BIAS
    """

    def is_train_ff():
      train_ff = False
      if self._hparams.optimize == 'ff' or self._hparams.optimize == 'both':
        train_ff = True
      return train_ff

    def is_train_fb():
      train_fb = False
      if self._hparams.optimize == 'fb' or self._hparams.optimize == 'both':
        train_fb = True
      return train_fb

    hidden_size = self._hparams.filters
    batch_hidden_shape = (self._hparams.batch_size, hidden_size)

    feedback_tensor = self._add_feedback()
    feedback_tensor = tf_print(feedback_tensor, "Feedback tensor: ", 100, mute=TF_DBUG_MUTE)

    # apply input mask to provide partial images
    input_shape = input_tensor.shape.as_list()
    sample_size = np.prod(input_shape[1:])
    input_mask = self._dual.add('input_mask', shape=[sample_size], default_value=1.0).add_pl(default=True)
    input_mask_reshaped = tf.reshape(input_mask, input_shape[1:])
    input_tensor = tf.multiply(input_tensor, input_mask_reshaped)   # apply shaped mask to each samples in batch

    # apply input gain (can be used to amplify or attenuate input)
    input_gain = self._dual.add('input_gain', shape=[1], default_value=1.0).add_pl(default=True)
    x_ff = tf.multiply(input_tensor, input_gain)

    self._dual.set_op('x_ff', x_ff)

    # Feed forward weighted sum
    # ------------------------------------------
    ws_ff = super()._build_weighted_sum(x_ff, add_bias=False, trainable=is_train_ff())

    # Feed back weighted sum
    # ------------------------------------------
    # ensure the feedback encoding has a different name so that it is a separate graph op
    hidden_name = self._hidden_name
    self._hidden_name = self._hidden_feedback_name
    ws_fb = super()._build_weighted_sum(feedback_tensor,
                                        add_bias=False,   # don't add bias for feedback
                                        trainable=is_train_fb())
    self._hidden_name = hidden_name

    # zero out single cell circular weights (i.e. cell 1 to cell 1), applied next batch before optimisation
    remove_circular = True
    if remove_circular:
      with tf.variable_scope(self._hidden_feedback_name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        weights_fb = tf.get_variable('kernel')
        mask = np.ones(weights_fb.get_shape(), dtype=np.float32)
        np.fill_diagonal(mask, 0)
        diagonal_mask = tf.convert_to_tensor(mask)
        weights_updated = tf.multiply(weights_fb, diagonal_mask)  # must be element-wise
        fb_weights_adjust_op = tf.assign(weights_fb, weights_updated, validate_shape=False, use_locking=True)
        self._dual.set_op('fb_weights_adjust_op', fb_weights_adjust_op)

    # Total weighted sum
    # ------------------------------------------
    ws = ws_fb + ws_ff

    # Non-linearity and masking
    # ------------------------------------------
    z_pre_mask, _ = activation_fn(ws, self._hparams.nonlinearity)

    # external masking
    name = hidden_name + '_masked'
    mask_pl = self._dual.add('mask', shape=batch_hidden_shape, default_value=1.0).add_pl()
    z = tf.multiply(z_pre_mask, mask_pl, name=name)

    # store reference to relevant ops for later use
    with tf.variable_scope(self._hidden_feedback_name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
      weights_fb = tf.get_variable('kernel')
      self._dual.set_op('weights_fb', weights_fb)
    with tf.variable_scope(self._hidden_name):
      weights_ff = tf.get_variable('kernel')
      self._dual.set_op('weights_ff', weights_ff)

      if self._hparams.use_bias:
        bias_ff = tf.get_variable('bias')
        self._dual.set_op('bias_ff', bias_ff)

    self._dual.set_op('ws_fb', ws_fb)
    self._dual.set_op('ws_ff', ws_ff)
    self._dual.set_op('ws', ws)
    self._dual.set_op('z', z)

    return z, z

  def _build_decoding(self, hidden_name, decoding_shape, filtered):
    decoding_ff = super()._build_decoding(hidden_name, decoding_shape, filtered)

    if self._hparams.reconstruct_fb_decoded:
      feedback = self._dual.get_op('feedback_input')
      decoding_fb = super()._build_decoding(self._hidden_feedback_name, feedback.get_shape().as_list(), filtered)
      self._dual.set_op('decoding_fb', decoding_fb)

    return decoding_ff

  def _build_optimizer(self):
    """Setup the training operations"""
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):

      self._optimizer = self._setup_optimizer()

      if self._hparams.loss_type != 'mse':
        raise NotImplementedError('Loss function not implemented: ' + str(self._hparams.loss_type))

      # ff fork - reconstruction of input
      # ------------------------------------
      x_ff = self._input_values
      y_ff = self.get_decoding_op()

      loss_ff = tf.losses.mean_squared_error(y_ff, x_ff)
      self._dual.set_op('loss_ff', loss_ff)

      # fb fork - use of hidden state
      # ------------------------------------
      if self._hparams.reconstruct_fb_decoded:
        x_fb = self._dual.get_op('feedback_dense')
        y_fb = self._dual.get_op('decoding_fb')  # reconstructed from sparse --> dense
        loss_fb = tf.losses.mean_squared_error(y_fb, x_fb)
      else:
        x_fb = self._dual.get_op('feedback_input')  # whatever was fed in, could be dense or sparse
        zs = self._dual.get_op('zs')  # sparse hidden state = filter(sum(ws_ff+ws_fb))
        loss_fb = tf.losses.mean_squared_error(zs, x_fb)
      self._dual.set_op('loss_fb', loss_fb)

      if self._hparams.optimize == 'ff':
        loss_op = loss_ff
      elif self._hparams.optimize == 'fb':
        loss_op = loss_fb
      elif self._hparams.optimize == 'both':
        loss_op = loss_ff + loss_fb
      else:
        raise NotImplementedError('Optimize type not implemented: ' + str(self._hparams.optimize))

      training_op = self._optimizer.minimize(loss_op, global_step=tf.train.get_or_create_global_step())

      self._dual.set_op('loss', loss_op)
      self._dual.set_op('training', training_op)

  ####################################################################
  # Summaries for the recursive loops

  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    super().build_summaries(batch_types, scope)  # build the summaries for all batch types

    self.build_recursive_summaries()  # and add graph structure for recursive summaries

  def build_recursive_summaries(self):
    """
    Same level as the `build_[batch_type]_summaries` methods
    """
    with tf.name_scope('recursive'):
      summaries = self._build_recursive_summaries()
      self._summary_recursive_op = tf.summary.merge(summaries)
      return self._summary_recursive_op

  def _build_recursive_summaries(self):
    """
    Build summaries to explicitly visualise 'recursive iterations' quantities
    """

    summaries = []
    max_outputs = 3

    feedback = self._dual.get_op('feedback_input')   # this has been normalised already at this stage
    encoding = self._dual.get_op('encoding')
    decoding_op = self.get_decoding_op()

    x_ff = self._dual.get_op('x_ff')
    ws_fb = self._dual.get_op('ws_fb')
    ws_ff = self._dual.get_op('ws_ff')
    z = self._dual.get_op('z')

    # ------------------ Images of Inputs -----------------
    summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)
    # raw input (input_values) + processed input (x_ff), which is direct input to the rsae
    input_reshape = tf.reshape(self._input_values, summary_input_shape)
    x_ff_reshape = tf.reshape(x_ff, summary_input_shape)
    input_preprocess = tf.concat([input_reshape, x_ff_reshape], axis=1)
    summaries.append(tf.summary.image('input_preprocess', input_preprocess, max_outputs=max_outputs))

    summaries.append(self._summary_hidden(feedback, 'feedback'))

    # ------------------- signals -------------------------
    with tf.name_scope('signals'):
      summaries.append(tf.summary.scalar('x', tf.reduce_sum(self._input_values)))
      summaries.append(tf.summary.scalar('x_ff', tf.reduce_sum(self._input_values)))
      summaries.append(tf.summary.scalar('y', tf.reduce_sum(encoding)))
      summaries.append(tf.summary.scalar('x_fb', tf.reduce_sum(feedback)))
      summaries.append(tf.summary.scalar('ws_fb', tf.reduce_sum(ws_fb)))
      summaries.append(tf.summary.scalar('ws_ff', tf.reduce_sum(ws_ff)))
      summaries.append(tf.summary.scalar('z', tf.reduce_sum(z)))

      # histograms of same things
      summaries.append(tf.summary.histogram('x', self._input_values))
      summaries.append(tf.summary.histogram('x_ff', x_ff))
      summaries.append(tf.summary.histogram('y', encoding))
      summaries.append(tf.summary.histogram('x_fb', feedback))
      summaries.append(tf.summary.histogram('ws_fb', ws_fb))
      summaries.append(tf.summary.histogram('ws_ff', ws_ff))
      summaries.append(tf.summary.histogram('z', z))

    # convergence of hidden binarised values
    with tf.name_scope('convergence'):
      # convergence - exploit fact that we know fb value is previous hidden state
      convergence_hidden = tf.reduce_sum(tf.abs(encoding - feedback))

      convergence_hidden_summary = tf.summary.scalar('y', convergence_hidden)
      summaries.append(convergence_hidden_summary)

      encoding_binary = tf.to_float(tf.greater(tf.abs(encoding), 0.0))
      feedback_binary = tf.to_float(tf.greater(tf.abs(feedback), 0.0))
      convergence_hidden_binary = tf.reduce_sum(tf.abs(encoding_binary - feedback_binary))

      convergence_hidden_binary_summary = tf.summary.scalar('y_binary', convergence_hidden_binary)
      summaries.append(convergence_hidden_binary_summary)

    # trainable variables
    with tf.name_scope('trainable_variables'):
      weights_ff = self._dual.get_op('weights_ff')
      weights_fb = self._dual.get_op('weights_fb')
      bias_ff = self._dual.get_op('bias_ff')
      bias_fb = self._dual.get_op('bias_fb')

      # scalars
      summaries.append(tf.summary.scalar('weights_ff', tf.reduce_sum(weights_ff)))
      summaries.append(tf.summary.scalar('weights_fb', tf.reduce_sum(weights_fb)))
      if bias_ff:
        summaries.append(tf.summary.scalar('bias_ff', tf.reduce_sum(bias_ff)))
      if bias_fb:
        summaries.append(tf.summary.scalar('bias_fb', tf.reduce_sum(bias_fb)))

      # histograms
      summaries.append(tf.summary.histogram('weights_ff', weights_ff))
      summaries.append(tf.summary.histogram('weights_fb', weights_fb))
      if bias_ff:
        summaries.append(tf.summary.histogram('bias_ff', bias_ff))
      if bias_fb:
        summaries.append(tf.summary.histogram('bias_fb', bias_fb))

    # raw input and reconstruction
    input_summary_reshape = tf.reshape(self._input_values, summary_input_shape)
    decoding_summary_reshape = tf.reshape(decoding_op, summary_input_shape)
    summary_reconstruction = tf.concat([input_summary_reshape, decoding_summary_reshape], axis=1)
    reconstruction_summary_op = tf.summary.image('z_reconstruction', summary_reconstruction, max_outputs=max_outputs)
    summaries.append(reconstruction_summary_op)

    # images of hidden components, hidden_feedback (previous hidden), hidden, both of which are post filtered
    summaries.append(self._summary_hidden(encoding, 'y'))
    summaries.append(self._summary_hidden(feedback, 'x_fb'))

    # images of weighted sums
    summaries.append(self._summary_hidden(ws_fb, 'ws_fb'))
    summaries.append(self._summary_hidden(ws_ff, 'ws_ff'))

    return summaries

  def write_recursive_summaries(self, step, writer, batch_type=None):
    """
    Write the summaries fetched into self._summary_recursive_values
    Ignore batch_type in this implementation
    """
    if self._summary_recursive_values is not None:
      writer.add_summary(self._summary_recursive_values, step)
      writer.flush()

  def add_fetches(self, fetches, batch_type='training'):
    super().add_fetches(fetches, batch_type)
    if self._summary_recursive_op is not None:
      fetches[self._name]['summaries_recursive'] = self._summary_recursive_op

  def set_fetches(self, fetched, batch_type='training'):
    super().set_fetches(fetched, batch_type)
    if self._summary_recursive_op is not None:
      self._summary_recursive_values = fetched[self._name]['summaries_recursive']

  # add some extra stuff to the main summaries as well
  def _build_summaries(self):

    summaries = super()._build_summaries()

    max_outputs = 3

    feedback = self._dual.get_op('feedback_input')  # this has been normalised already at this stage
    encoding = self._dual.get_op('encoding')

    x_ff = self._dual.get_op('x_ff')
    ws_fb = self._dual.get_op('ws_fb')
    ws_ff = self._dual.get_op('ws_ff')
    z = self._dual.get_op('z')

    weights_ff = self._dual.get_op('weights_ff')
    weights_fb = self._dual.get_op('weights_fb')
    bias_ff = self._dual.get_op('bias_ff')
    bias_fb = self._dual.get_op('bias_fb')

    # add two components of loss
    loss_ff = self._dual.get_op('loss_ff')
    loss_fb = self._dual.get_op('loss_fb')

    loss_ff_summary = tf.summary.scalar('loss_ff', loss_ff)
    loss_fb_summary = tf.summary.scalar('loss_fb', loss_fb)
    summaries.append(loss_ff_summary)
    summaries.append(loss_fb_summary)

    # images
    # -----------------------------------------------------

    # feedback reconstruction
    if self._hparams.reconstruct_fb_decoded:
      x = self._dual.get_op('feedback_dense')
      y = self._dual.get_op('decoding_fb')
    else:
      x = self._dual.get_op('feedback_input')
      y = self._dual.get_op('encoding')  # sparse hidden state

    hidden_shape_4d = self._hidden_image_summary_shape()  # [batches, height=1, width=filters, 1]
    x_image_shape = tf.reshape(x, hidden_shape_4d)
    y_image_shape = tf.reshape(y, hidden_shape_4d)
    fb_reconstruction = tf.concat([x_image_shape, y_image_shape], axis=1)
    fb_reconstruction_summary = tf.summary.image('fb_reconstruction', fb_reconstruction, max_outputs=max_outputs)
    summaries.append(fb_reconstruction_summary)

    if self._hparams.summary_long:

      # summary_input_shape = image_utils.get_image_summary_shape(self._input_values)

      # # raw input (input_values) + processed input (x_ff), which is direct input to the rsae
      # input_reshape = tf.reshape(self._input_values, summary_input_shape)
      # x_ff_reshape = tf.reshape(x_ff, summary_input_shape)
      # input_preprocess = tf.concat([input_reshape, x_ff_reshape], axis=1)
      # summaries.append(tf.summary.image('input_preprocess', input_preprocess, max_outputs=max_outputs))

      with tf.name_scope('trainable-variables'):
        if bias_ff:
          add_square_as_square(summaries, bias_ff, 'bias_ff')
        if bias_fb:
          add_square_as_square(summaries, bias_fb, 'bias_fb')
        add_square_as_square(summaries, weights_ff, 'weights_ff')
        add_square_as_square(summaries, weights_fb, 'weights_fb')

      with tf.name_scope('signals'):
        # histograms of same things
        summaries.append(tf.summary.histogram('x', self._input_values))
        summaries.append(tf.summary.histogram('x_ff', x_ff))
        summaries.append(tf.summary.histogram('y', encoding))
        summaries.append(tf.summary.histogram('x_fb', feedback))
        summaries.append(tf.summary.histogram('ws_fb', ws_fb))
        summaries.append(tf.summary.histogram('ws_ff', ws_ff))
        summaries.append(tf.summary.histogram('z', z))

      # histograms
      add_weights_histograms = True
      if add_weights_histograms:
        with tf.name_scope('trainable-variables'):
          summaries.append(tf.summary.histogram('weights_ff', weights_ff))
          summaries.append(tf.summary.histogram('weights_fb', weights_fb))
          if bias_ff:
            summaries.append(tf.summary.histogram('bias_ff', bias_ff))
          if bias_fb:
            summaries.append(tf.summary.histogram('bias_fb', bias_fb))

    return summaries

