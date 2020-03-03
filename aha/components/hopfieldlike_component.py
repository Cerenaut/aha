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

"""HopfieldlikeComponent class."""

import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


from pagi.utils.dual import DualData
from pagi.utils import image_utils, generic_utils, tf_utils
from pagi.utils.image_utils import add_square_as_square, square_image_shape_from_1d, add_op_images
from pagi.utils.layer_utils import activation_fn, type_activation_fn

from pagi.components.component import Component
from pagi.components.summarize_levels import SummarizeLevels

from aha.utils.generic_utils import build_kernel_initializer, normalize_minmax, print_minmax

########################################################################################

# The following methods convert signals between spaces
# i.e. from real values between 0,1 to binary values of values -1 or 1
# The current implementations are for ONE setup that we are currently using, they are not flexible
# If we are to change the DG outputs, we'll need to change these or parameterise
#
# PC space = the memory to memorise
# PC cue space = cue to use for pr (can be real valued)
#
# Assumes:
# DG =  [0, 1] real valued, sparse (i.e. there are definitely units with 0 output)
# PC =  [-1, 1] binary valued
# PC_cue = [-1, 1] real valued


def pc_to_unit(tensor):
  """
  From b[-1,1] to [0,1]
  This implementation only works assuming tensor is binary
  """
  # if >0, will be 1, if ==-1 (<0), will be 0
  tensor = tf.to_float(tf.greater(tensor, 0))   # 1.0(True) where =curr_min, 0.(False) where !=curr_min
  return tensor


def unit_to_pc_linear(tensor):
  """Input assumed to be unit range. Linearly scales to -1 <= x <= 1"""
  result = (tensor * 2.0) - 1.0  # Theoretical range limits -1 : 1
  return result


def unit_to_pc_sparse(tensor):
  """ From b[0,1] to b[-1,1] or [0,1] to r[-1,1] """
  tensor, _ = tf_utils.tf_set_min(tensor, None, tgt_min=-1, current_min=0)  # set 0's to -1, so that the range is -1, 1
  return tensor


def get_pc_topk_shift(tensor, sparsity):
  """Input tensor must be batch of vectors.
  Returns a vector per batch sample of the shift required to make Hopfield converge.
  Assumes knowledge of Hopfield fixed sparsity."""

  # Intuition: The output distribution must straddle the zero point to make hopfield work.
  # These are the values that should be positive.
  tensor_shape = tensor.get_shape().as_list()
  batch_size = tensor_shape[0]
  num_features = tensor_shape[1]
  cue_top_k_mask = tf_utils.tf_build_top_k_mask_op(input_tensor=tensor,
                                                   k=int(sparsity+1),  # NOTE: k+1th elem := 0
                                                   batch_size=batch_size,
                                                   input_area=num_features)
  y = tensor
  # Worked example:
  # e.g. k = 2
  # 0, 0.1, 0.3, 0.5   y
  # 0  0    1    1     mask
  # 1-y:
  # 1  0.9  0.7  0.5   y_inv
  # * mask
  # 0, 0  , 0.7, 0.5   y_inv_masked
  # max: 0.7
  # 1-max: 0.3
  y_inv = 1.0 - y
  y_inv_masked = y_inv * cue_top_k_mask
  y_inv_masked_max = tf.reduce_max(y_inv_masked, axis=1)  # max per batch sample
  y_masked_min = 1.0 - y_inv_masked_max

  # convert this to tanh range
  # cue_tanh_min:  -0.5 -0.1 0.0  0.1  0.5
  # 0-x            +0.5 +0.1 0.0 -0.1 -0.5
  # so e.g.
  #             -0.5 + 0.5 = 0
  #              0.5 + -0.5 = 0  this bit value has become zero.
  cue_tanh_masked_min = (y_masked_min * 2.0) - 1.0  # scale these values
  shift = tf.expand_dims(0.0 - cue_tanh_masked_min, 1)
  return shift


def dg_to_pc(tensor):
  """ From sparse r[0,1] to b[-1,1]"""
  tensor = tf.to_float(tf.greater(tensor, 0.0))
  tensor, _ = tf_utils.tf_set_min(tensor, None, -1, current_min=0)  # set 0's to -1, so that the range is -1, 1
  return tensor


# def dg_to_pc_numpy(arr):
#
#   arr = np.greater(arr, 0.0).astype(float)
#
#   minval = -1
#   source_zeros = np.equal(arr, 0).astype(float)   # 1.0(True) where =0, 0.(False) where !=0
#   minvals_inplace = minval * source_zeros
#   target = arr + minvals_inplace
#
#   return target

########################################################################################


class HopfieldlikeComponent(Component):
  """
  Hopfield net inspired component. Main ideas of Hopfield network are implemented.
  The training differs though, this version uses gradient descent to minimise the diff
  between input and output calculated simply by one pass through, no activation function.
  i.e.    Y = MX,   loss = Y - X

  pr PATH
  If an input_cue is specified when build() is called, then `use_input_cue mode=True`,
  and the Hopfield learns to map an external to internal cue for retrieval.
  This can be done via a pseudoinverse or NN (set by internal static constant)
  Pseudoinverse is not recommended as it is symmetrical, which is pathological, but it works ok for simple cases.

  Batch types are:
  - training  : memorise the samples in a batch in the fb weights, & for `use_input_cue mode`, learn to map external
                cue `x_cue` to internal cue `z_cue` which is used for retrieval.
  - encoding :  'retrieval' but use 'encoding' for compatibility other components. Recursive steps to produce output.

  Terminology:

  x_ext = external input (from DG). Used to memorise batch.

  x_cue = external cue input (from EC). Any dimensions. Mapped to z_cue which is used for retrieval.
  z_cue = output of cue mapping (x_cue to Hopfield dimensions)

  x_fb = feedback = y(t-1), recursive iterations get Hopfield converging on basins of attraction

  y = output of net at time t

  WARNINGS
  # Builds ONE 'recursive' summary subgraph (i.e. these may be produced in the context of any batch type)

  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        batch_size=1,
        learning_rate=0.0001,
        optimizer='adam',
        momentum=0.9,
        momentum_nesterov=False,
        use_feedback=True,
        memorise_method='pinv',       # pinv = pseudoinverse, otherwise tf optimisation
        nonlinearity='tanh',
        update_n_neurons=-1,          # number of neurons to update each iteration, -1 = all
        gain=2.7,                     # applied to weights during iterations

        pr_type='pinv',               # 'nn' or 'pinv': type of cue mapping in pr path
        pm_type='none',               # 'none' or 'nn': map stable PC patterns back to EC
        pm_raw_type='none',           # 'none' or 'nn': map stable PC patterns back to VC input (image space)
        pm_l1_size=100,               # hidden layer of PM path (pattern mapping)
        pm_raw_l1_size=100,
        pm_raw_l2_regularizer=0.0,
        pm_raw_nonlinearity='leaky_relu',
        pm_noise_type='s',            # 's' for salt, 'sp' for salt + pepper
        pm_train_with_noise=0.0,
        pm_train_with_noise_pp=0.0,

        cue_nn_train_dropout_keep_prob=1.0,  # 1.0=off: dropout in nn that learns the cue from EC
        cue_nn_test_with_noise=0.0,       # 0.0=off: noise to EC for testing generalisation of learning cue with nn
        cue_nn_train_with_noise=0.0,      # 0.0=off: noise to EC for testing generalisation of learning cue with nn
        cue_nn_train_with_noise_pp=0.0,
        cue_nn_test_with_noise_pp=0.0,
        cue_nn_label_sparsity=10,
        cue_nn_hidden_size=500,
        cue_nn_sparsity_boost=1.2,    # let more bits through, can tolerate false positives better
        cue_nn_non_linearity='sigmoid',
        cue_nn_last_layer='softmax_ce',  # 'softmax_ce', 'sigmoid_mse', 'relu_mse', 'linear_mse'
        cue_nn_gain=1.0,  # 'softmax_ce', 'sigmoid_mse', 'relu_mse', 'linear_mse'
        cue_nn_sum_norm=10.0,
        cue_nn_softmax=False,
        cue_nn_sparsen=False,
        cue_nn_l2_regularizer=0.0,

        summarize_level=SummarizeLevels.ALL.value,
        max_outputs=3
    )

  def __init__(self):
    self._name = None
    self._hidden_name = None
    self._hparams = None
    self._dual = None
    self._input_summary_shape = None
    self._input_values = None
    self._input_cue = None
    self._input_cue_raw = None
    self._use_input_cue = None
    self._use_pm = False
    self._use_pm_raw = False
    self._summary_values = None
    self._summary_recursive_values = None
    self._input_size = None
    self._input_values_shape = None
    self._batch_type = None

    self._debug_input_cue = False

  def reset(self):
    self._dual.get('y').set_values_to(0.0)

    loss = self._dual.get('loss_memorise')
    loss.set_values_to(0.0)

    if self._use_input_cue:
      pr_loss = self._dual.get('pr_loss')
      pr_loss.set_values_to(0.0)

  # -------------------------------------------------------
  # used for playing around with pinv

  def get_cue_target(self):

    # how we did it before
    x = self._dual.get_op('x_ext')          # DG - primary, intended output (already set to pc space)

    # using raw so that it learns based on signal (0, 1) i.e. no negative weights. We'll map it afterwards.
    # x = self._dual.get_op('x_ext_raw')          # DG - primary, intended output

    return x

  def modify_pr_out(self, x):
    # return dg_to_pc(x)
    return x

  # -------------------------------------------------------

  @property
  def name(self):
    return self._name

  @property
  def use_input_cue(self):
    return self._use_input_cue

  @property
  def use_pm(self):
    return self._use_pm

  @property
  def use_pm_raw(self):
    return self._use_pm_raw

  @property
  def use_inhibition(self):
    return True

  def get_loss(self):
    """Loss from memorisation of samples in fb weights """
    return self._dual.get_values('loss_memorise')

  def get_loss_pr(self, default=0):
    """Loss from mapping in pr branch (from EC external cue to PC internal cue"""
    loss = self._dual.get_values('pr_loss_mismatch')
    if loss is None:
      loss = default
    return loss

  def get_loss_pr_range(self):
    """Minimum possible loss returned from get_pr_loss()"""
    sparsity = self._hparams.cue_nn_label_sparsity
    boost = self._hparams.cue_nn_sparsity_boost
    min = sparsity - int(sparsity * boost)   # they are exactly the same, except for the additional allowed bits
    max = sparsity + int(sparsity * boost)   # no overlap in the label and prediction
    return min, max

  def get_losses_pm(self, default=0):
    loss = self._dual.get_values('pm_loss')
    loss_raw = self._dual.get_values('pm_loss_raw')

    if loss is None:
      loss = default

    if loss_raw is None:
      loss_raw = default

    return loss, loss_raw

  def get_dual(self):
    return self._dual

  def get_decoding(self):
    """For consistency with other components, 'decoding' is output, y. Reshaped to input dimensions."""
    return self._dual.get_values('decoding')

  def get_decoding_op(self):
    """For consistency with other components, 'decoding' is output, y. Reshaped to input dimensions."""
    return self._dual.get_op('decoding')

  def get_input(self, batch_type):
    """
    batch_type: 'training' = memorization, 'encoding' = retrieval

    For memorisation, input = x_ext
    For retrieval, input depends on whether PR is in use
    """

    if batch_type == 'training':
      return self._dual.get_values('x_ext')
    else:
      if self._use_input_cue:
        return self._dual.get_values('pr_out')
      else:
        return self._dual.get_values('x_direct')

  def get_ec_out_raw_op(self):
    return self._dual.get_op('ec_out_raw')

  def get_ec_out_raw(self):
    return self._dual.get_values('ec_out_raw')

  def update_feed_dict(self, feed_dict, batch_type='training'):
    if batch_type == 'training':
      self.update_training_dict(feed_dict)
    if batch_type == 'encoding':
      self.update_encoding_dict(feed_dict)

  def _update_dict_fb(self, feed_dict):
    # set feedback from previous y output
    x_next = self._dual.get('y').get_values()
    x_fb = self._dual.get_pl('x_fb')
    feed_dict.update({
        x_fb: x_next
    })

  def update_feed_dict_input_gain_pl(self, feed_dict, gain):
    input_gain_pl = self._dual.get('input_gain').get_pl()
    feed_dict.update({
        input_gain_pl: [gain]
    })

  def add_fetches(self, fetches, batch_type='training'):
    if batch_type == 'training':
      self.add_training_fetches(fetches)
    if batch_type == 'encoding':
      self.add_encoding_fetches(fetches)

    summary_op = self._dual.get_op(generic_utils.summary_name(batch_type))
    if summary_op is not None:
      fetches[self._name]['summaries'] = summary_op

    summary_op = self._dual.get_op(generic_utils.summary_name('recursive'))
    if summary_op is not None:
      fetches[self._name]['summaries_recursive'] = summary_op

  def set_fetches(self, fetched, batch_type='training'):
    if batch_type == 'training':
      self.set_training_fetches(fetched)
    if batch_type == 'encoding':
      self.set_encoding_fetches(fetched)

    summary_op = self._dual.get_op(generic_utils.summary_name(batch_type))
    if summary_op is not None:
      self._summary_values = fetched[self._name]['summaries']

    summary_recursive_op = self._dual.get_op(generic_utils.summary_name('recursive'))
    if summary_recursive_op is not None:
      self._summary_recursive_values = fetched[self._name]['summaries_recursive']

  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:

        # build 'batch_type' summary subgraph
        with tf.name_scope(batch_type):
          summaries = self._build_summaries(batch_type)
          if summaries and len(summaries) > 0:
            self._dual.set_op(generic_utils.summary_name(batch_type), tf.summary.merge(summaries))

      # WARNING: Build ONE 'recursive' summary subgraph (i.e. these may be produced in the context of any batch type)
      with tf.name_scope('recursive'):
        summaries = self._build_recursive_summaries()
        if len(summaries) > 0:
          self._dual.set_op(generic_utils.summary_name('recursive'), tf.summary.merge(summaries))

  def write_summaries(self, step, writer, batch_type='training'):
    """Write the summaries fetched into _summary_values"""
    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)
      writer.flush()

  def write_recursive_summaries(self, step, writer, batch_type='training'):
    """
    Only write summaries for encoding batch_type (retrieval)
    """
    if batch_type == 'encoding':
      if self._summary_recursive_values is not None:
        writer.add_summary(self._summary_recursive_values, step)
        writer.flush()

# ---------------- build methods

  def build(self, input_values, input_summary_shape, hparams, name, input_cue=None, input_cue_raw=None):
    """Builds the network and optimiser."""
    self._input_values = input_values
    self._input_summary_shape = input_summary_shape
    self._hparams = hparams
    self._name = name

    self._input_values_shape = self._input_values.get_shape().as_list()
    self._input_size = np.prod(self._input_values_shape[1:])

    if self._debug_input_cue:
      input_cue = input_values

    # if input_cue is provided, then set flag to build PR
    if input_cue is not None:
      self._input_cue = input_cue
      self._use_input_cue = True
    else:
      self._use_input_cue = False

    # if hyperparam specifies and input_cue or input_cue_raw provided, set flag to build PM
    if self._hparams.pm_type != 'none' and input_cue is not None:
      self._use_pm = True

    if self._hparams.pm_raw_type != 'none' and input_cue_raw is not None:
      self._input_cue_raw = input_cue_raw
      self._use_pm_raw = True

    self._dual = DualData(self._name)

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

      self._batch_type = tf.placeholder_with_default(input='training', shape=[], name='batch_type')

      # 0) setup inputs for other build methods
      # ---------------------------------------------------
      self._setup_inputs()    # sets: x_cue, x_ext, x_fb

      self._random_recall = self._dual.add('random_recall',
                                           shape=[],
                                           default_value='').add_pl(default=True, dtype=tf.string)

      # 1) build cue mapping for retrieval (if relevant)
      # ---------------------------------------------------
      if self._use_input_cue:
        if self.use_nn_in_pr_path():
          self._build_pr_nn()  # sets: vc_to_pc, loss_learn_cue, uses: x_cue, x_ext
        else:
          self._build_pr_pinv(self._input_size)  # sets: w_p, z_cue

      # 2) build core Hopfield for retrieval - recursive retrieval network
      # ---------------------------------------------------
      self._build_retrieval()           # uses: x_cue, x_ext, x_fb, vc_to_pc [with cue_pinv: w_p, z_cue]

      # 3) build cue retrieval memorisation for pinv variant
      #    note: must be after build_retrieval()
      # ---------------------------------------------------
      if self._use_input_cue:
        if not self.use_nn_in_pr_path():
          self._build_pr_pinv_memorise()  # uses: x_ext, x_cue

      # 4) build Pattern Mapping (PC patterns to corresponding EC)
      # ---------------------------------------------------
      if self._use_pm or self._use_pm_raw:
        self._build_pm()

      # 5) build Hopfield fb weights - memorisation of x_ext (DG) inputs
      # ---------------------------------------------------
      if self._is_pinv():
        self._build_memorise_pinv()

      self.reset()

  def _setup_inputs(self):
    """Prepare external input by reshaping and optionally applying an input gain."""
    input_values_shape = self._input_values.get_shape().as_list()
    input_size = np.prod(input_values_shape[1:])
    input_shape = [self._hparams.batch_size, input_size]

    # external input (from DG)
    x_ext = tf.reshape(self._input_values, input_shape)

    self._dual.set_op('x_ext_raw', x_ext)

    x_ext = dg_to_pc(x_ext)

    # apply input gain (can be used to amplify or attenuate external input)
    input_gain = self._dual.add('input_gain', shape=[1], default_value=1.0).add_pl(default=True)
    x_ext = tf.multiply(x_ext, input_gain)
    self._dual.set_op('x_ext', x_ext)

    # placeholder for getting feedback signal
    self._dual.add('x_fb', shape=input_shape, default_value=0.0).add_pl(default=True)

    # input cue (from EC)
    if self._use_input_cue:
      input_cue_shape = self._input_cue.get_shape().as_list()
      input_cue_size = np.prod(input_cue_shape[1:])
      x_cue_shape = [self._hparams.batch_size, input_cue_size]
      x_cue = tf.reshape(self._input_cue, x_cue_shape)

      self._dual.set_op('x_cue', x_cue)

  def _w_variable(self, shape, trainable=False):
    w_default = 0.01
    w_initializer = w_default * tf.random_uniform(shape)

    if not self._is_pinv() and not self._is_pinv_hybrid():
      trainable = True

    # Apply a constraint to zero out single cell circular weights (i.e. cell 1 to cell 1)
    return tf.get_variable(name='w', initializer=w_initializer, constraint=self._remove_diagonal,
                           trainable=trainable)

  def _neuron_update(self, input_size, x_fb, y_potential):

    # compute mask
    length_update = self._hparams.update_n_neurons
    if length_update == -1:
      length_update = input_size

    seed_np = np.ones(input_size)
    seed_np[length_update:] = 0
    seed = tf.convert_to_tensor(seed_np, dtype=tf.float32)
    mask = tf.random_shuffle(seed)      # 1 in the bits to update

    # apply masked update
    y_update = tf.multiply(y_potential, mask)    # zero out the non-update neurons
    y_purge_mask = 1 - mask
    y_temp = tf.multiply(y_purge_mask, x_fb)     # x_fb is baseline, now zero out the neuron to be updated
    y = y_temp + y_update                        # apply update selectively on that neuron

    return y

  def _build_retrieval(self):
    """
    Initialises variables and builds the Hopfield-like network.

    Retrieve with x_direct and x_fb

    When `_use_input_cue` enabled:
      - x_direct = z_cue (output of mapping from external cue (EC) to internal cue)
    Else:
      - x_direct = x_ext  (DG output)

    """

    input_values_shape = self._input_values_shape
    input_size = self._input_size

    # create variables
    w = self._w_variable(shape=[input_size, input_size])

    # setup network
    if self._use_input_cue:
      x_direct = self._dual.get_op('z_cue')
    else:
      x_ext = self._dual.get_op('x_ext')
      x_direct = x_ext  # no weights, one-to-one mapping, so they are the same

    pc_noise = self._dual.add('pc_noise',
                              shape=x_direct.shape,
                              default_value=0.0).add_pl(default=True, dtype=x_direct.dtype)

    # Swap 'x_direct' during random recall at PC
    x_direct = tf.cond(tf.equal(self._random_recall, 'pc'), lambda: pc_noise, lambda: x_direct)

    x_fb = self._dual.get_pl('x_fb')
    z = tf.matmul(x_fb, w) + x_direct  # weighted sum + bias
    y_potential, _ = activation_fn(self._hparams.gain * z, self._hparams.nonlinearity)  # non-linearity

    # only update the relevant neurons
    y = self._neuron_update(input_size, x_fb, y_potential)

    # calculate Hopfield Energy
    e = -0.5 * tf.matmul(tf.matmul(y, w), tf.transpose(y)) - tf.matmul(y, tf.transpose(x_direct))

    # 'decoding' for output in same dimensions as input, and for consistency with other components
    y_reshaped = tf.reshape(y, input_values_shape)

    # Normalize the decoding output
    y_reshaped = normalize_minmax(y_reshaped)

    # remember values for later use
    self._dual.set_op('w', w)
    self._dual.set_op('y', y)
    self._dual.set_op('decoding', y_reshaped)
    self._dual.set_op('e', e)
    self._dual.set_op('x_direct', x_direct)

    return y

  def _build_pm(self):
    """Preprocess the inputs and build the pattern mapping components."""

    # map to input
    # pc_out = self._dual.get_op('y')  # output of Hopfield (PC)
    pc_out = self._dual.get_op('decoding')  # output of Hopfield (PC)
    # pc_out = normalize_minmax(pc_out)

    pc_target = self._dual.get_op('pr_target')

    x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                   lambda: pc_target,       # training
                   lambda: pc_out)          # encoding

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

    # Build PM
    # --------------------------------------------------------------------------
    if self.use_pm:
      ec_in = self._input_cue
      output_nonlinearity = type_activation_fn('leaky_relu')
      ec_out = self._build_pm_core(x=x_nn, target=ec_in,
                                   l1_size=self._hparams.pm_l1_size,
                                   non_linearity1=tf.nn.leaky_relu,
                                   non_linearity2=output_nonlinearity,
                                   loss_fn=tf.losses.mean_squared_error)
      self._dual.set_op('ec_out', ec_out)

    if self._use_pm_raw:
      ec_in = self._input_cue_raw
      output_nonlinearity = type_activation_fn(self._hparams.pm_raw_nonlinearity)
      ec_out_raw = self._build_pm_core(x=x_nn, target=ec_in,
                                       l1_size=self._hparams.pm_raw_l1_size,
                                       non_linearity1=tf.nn.leaky_relu,
                                       non_linearity2=output_nonlinearity,
                                       loss_fn=tf.losses.mean_squared_error,
                                       name_suffix="_raw")
      self._dual.set_op('ec_out_raw', ec_out_raw)

  def _build_pm_core(self, x, target, l1_size, non_linearity1, non_linearity2, loss_fn, name_suffix=""):
    """Build the layers of the PM network, with optional L2 regularization."""
    target_shape = target.get_shape().as_list()
    target_size = np.prod(target_shape[1:])
    l2_size = target_size

    use_bias = True

    weights = []
    scope = 'pm' + name_suffix
    with tf.variable_scope(scope):
      y1_layer = tf.layers.Dense(units=l1_size, activation=non_linearity1, use_bias=use_bias,
                                 kernel_initializer=build_kernel_initializer('xavier'))
      y1 = y1_layer(x)

      f_layer = tf.layers.Dense(units=l2_size, activation=non_linearity2, use_bias=use_bias,
                                kernel_initializer=build_kernel_initializer('xavier'))
      f = f_layer(y1)

      weights.append(y1_layer.weights[0])
      weights.append(f_layer.weights[0])

      if use_bias:
        weights.append(y1_layer.weights[1])
        weights.append(f_layer.weights[1])

      y = tf.stop_gradient(f)   # ensure gradients don't leak into other nn's in PC

    target_flat = tf.reshape(target, shape=[-1, target_size])
    loss = loss_fn(f, target_flat)
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

  def _build_pr_pinv(self, input_size):
    input_cue_shape = self._input_cue.get_shape().as_list()
    input_cue_size = np.prod(input_cue_shape[1:])
    w_p = tf.get_variable(name='w_p', shape=(input_cue_size, input_size), trainable=False)
    self._dual.set_op('w_p', w_p)

    x_cue = self._dual.get_op('x_cue')
    z_cue = tf.matmul(x_cue, w_p)

    z_cue = self.modify_pr_out(z_cue)

    self._dual.set_op('z_cue', z_cue)

    return z_cue

  def _build_memorise_pinv(self):
    """Pseudoinverse-based optimisation."""
    input_values_shape = self._input_values.get_shape().as_list()
    input_size = np.prod(input_values_shape[1:])

    x = self._dual.get_op('x_ext')

    w_ref = self._dual.get_op('w')

    batches = input_values_shape[0]
    x_matrix = tf.reshape(x, [1, batches, input_size])  # 1 matrix of x vol vecs (expressed as 1 batch)
    xinv = tfp.math.pinv(x_matrix, rcond=None, validate_args=False, name=None)  # this is XT-1 (transposed already)
    w_batches = tf.matmul(xinv, x_matrix)
    w_val = tf.reshape(w_batches, [input_size, input_size])   # strip out the batch dimension
    w_val = self._remove_diagonal(w_val)  # remove self-connections (also a constraint if using gradient training)
    w = tf.assign(w_ref, w_val, name='w_assign')

    y_memorise = tf.matmul(x, w)
    loss_memorise = tf.reduce_sum(tf.square(x - y_memorise))

    self._dual.set_op('y_memorise', y_memorise)
    self._dual.set_op('loss_memorise', loss_memorise)

  def _build_pr_nn(self):
    """
    Teach a NN to transform x_cue to x (=z_cue)
    From the NN perspective:

    x_nn = x_cue
    t_nn = x  (target)

    """

    # 0) nn params
    # ------------------------------------
    sparsity = self._hparams.cue_nn_label_sparsity
    pr_sparsity_boost = self._hparams.cue_nn_sparsity_boost
    non_linearity = self._hparams.cue_nn_non_linearity
    hidden_size = self._hparams.cue_nn_hidden_size

    # 1) organise inputs to network
    # ------------------------------------
    x_ext = self._dual.get_op('x_ext')  # DG - primary, converted to PC space (-1<=n<=1)

    t_nn = pc_to_unit(x_ext)  # This means unit range, sparse
    self._dual.set_op('pr_target', t_nn)

    t_nn_shape = x_ext.get_shape().as_list()
    t_nn_size = np.prod(t_nn_shape[1:])

    x_nn = self._dual.get_op('x_cue')  # EC - secondary     : vc output
    x_nn_shape = x_nn.get_shape().as_list()
    x_nn_size = np.prod(x_nn_shape[1:])

    pr_noise = self._dual.add('pr_noise',
                              shape=x_nn.shape,
                              default_value=0.0).add_pl(default=True, dtype=x_nn.dtype)

    inhibition = self._dual.add('inhibition',
                                shape=x_nn.shape,
                                default_value=0.0).add_pl(default=True, dtype=x_nn.dtype)

    use_inhibition_pl = self._dual.add('use_inhibition',
                                       shape=[],
                                       default_value=False).add_pl(default=True, dtype=tf.bool)

    if self.use_inhibition:
      decay = 0.5
      inhibition_decayed = decay * inhibition + (1 - decay) * x_nn
      self._dual.set_op('inhibition', inhibition_decayed)

      inhibition_noise = pr_noise + inhibition_decayed
      random_cue = tf.cond(tf.equal(use_inhibition_pl, True), lambda: inhibition_noise, lambda: pr_noise)
    else:
      random_cue = pr_noise

    # Swap 'x_nn' during random recall
    x_nn = tf.cond(tf.equal(self._random_recall, 'pr'), lambda: random_cue, lambda: x_nn)

    x_nn = x_nn

    # 2) build the network
    # ------------------------------------

    # apply noise at train and/or test time, to regularise / test generalisation
    x_nn = tf.cond(tf.equal(self._batch_type, 'encoding'),
                   lambda: image_utils.add_image_salt_noise_flat(x_nn, None,
                                                                 noise_val=self._hparams.cue_nn_test_with_noise,
                                                                 noise_factor=self._hparams.cue_nn_test_with_noise_pp),
                   lambda: x_nn)

    x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                   lambda: image_utils.add_image_salt_noise_flat(x_nn, None,
                                                                 noise_val=self._hparams.cue_nn_train_with_noise,
                                                                 noise_factor=self._hparams.cue_nn_train_with_noise_pp),
                   lambda: x_nn)

    # apply dropout during training
    keep_prob = self._hparams.cue_nn_train_dropout_keep_prob
    x_nn = tf.cond(tf.equal(self._batch_type, 'training'),
                   lambda: tf.nn.dropout(x_nn, keep_prob),
                   lambda: x_nn)

    self._dual.set_op('x_pr_memorise', x_nn)    # input to the pr path nn

    # Hidden layer[s]
    weights = []

    if hidden_size > 0:

      kernel_initializer = build_kernel_initializer('xavier')

      # hidden_out = tf.layers.dense(inputs=x_nn, units=hidden_size, activation=type_activation_fn(non_linearity),
      #                              name="cue_nn_hidden")
      layer_hidden = tf.layers.Dense(units=hidden_size,
                                     activation=type_activation_fn(non_linearity),
                                     name="cue_nn_hidden",
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

    # Final layer - no dropout (because we don't want to damage the output signal, there's no benefit)
    # No nonlinearity (yet)

    kernel_initializer = build_kernel_initializer('xavier')

    layer_out = tf.layers.Dense(units=t_nn_size,
                                name="cue_nn_logits",
                                kernel_initializer=kernel_initializer)    # units = number of logits
    logits = layer_out(hidden_out)
    weights.append(layer_out.weights[0])
    weights.append(layer_out.weights[1])
    #logits = tf.layers.dense(inputs=hidden_out, units=t_nn_size, name="cue_nn_logits")    # units = number of logits

    last_layer = self._hparams.cue_nn_last_layer

    if last_layer == 'relu_softmax_ce':
      f = tf.nn.relu(logits)
      probs = tf.nn.softmax(f)
      loss = tf.losses.sigmoid_cross_entropy(t_nn, f)
      f = probs * sparsity  # adjust output magnitudes: sums to 1, restore magnitudes of individual bits to range=(0,1)
    elif last_layer == 'sigmoid_softmax_ce':
      f = tf.nn.sigmoid(logits)
      probs = tf.nn.softmax(f)
      loss = tf.losses.sigmoid_cross_entropy(t_nn, f)
      f = probs * sparsity  # adjust output magnitudes: sums to 1, restore magnitudes of individual bits to range=(0,1)
    elif last_layer == 'softmax_ce':  # Original mode
      probs = tf.nn.softmax(logits)
      loss = tf.losses.sigmoid_cross_entropy(t_nn, logits)
      f = probs * sparsity  # adjust output magnitudes: sums to 1, restore magnitudes of individual bits to range=(0,1)
    elif last_layer == 'sigmoid_ce':  # Dave's mode. Treat each output px as a class. problem
      f = tf.nn.sigmoid(logits)  # Unit range
      loss = tf.losses.sigmoid_cross_entropy(t_nn, logits)
    elif last_layer == 'sigmoid_mse':  # Alternate training regime to try, same output
      #y = logits #tf.nn.sigmoid(logits) WORKS
      f = tf.nn.sigmoid(logits)
      loss = tf.losses.mean_squared_error(t_nn, f)
    elif last_layer == 'lrelu_mse':  # Alternate training regime to try, same output
      f = tf.nn.leaky_relu(logits)
      loss = tf.losses.mean_squared_error(t_nn, f)
    else:
      raise RuntimeError("cue_nn_last_layer hparam option '{}' not implemented".format(last_layer))

    y = tf.stop_gradient(f)

    if self._hparams.cue_nn_l2_regularizer > 0.0:
      all_losses = [loss]

      for weight in weights:
        weight_loss = tf.nn.l2_loss(weight)
        weight_loss_sum = tf.reduce_sum(weight_loss)
        weight_loss_scaled = weight_loss_sum * self._hparams.cue_nn_l2_regularizer
        all_losses.append(weight_loss_scaled)

      all_losses_op = tf.add_n(all_losses, name='total_pr_loss')
      self._build_optimizer(all_losses_op, 'training_pr', scope='pr')
    else:
      self._build_optimizer(loss, 'training_pr', scope='pr')

    # Swap 'y' during replay
    # y = tf.cond(tf.equal(replay, True), lambda: replay_input, lambda: y)

    self._dual.set_op('pr_probs', y)    # badly named for historical reasons

    if (last_layer == 'sigmoid_ce') or (last_layer == 'sigmoid_mse') or (last_layer == 'lrelu_mse'):  # New modes
      logging.info('PR New last layer pathway enabled.')

      # Clip
      y = tf.clip_by_value(y, 0.0, 1.0)

      # Sparsen
      if self._hparams.cue_nn_sparsen is True:
        k_pr = int(sparsity * pr_sparsity_boost)
        logging.info('PR Sparsen enabled k=' + str(k_pr))
        mask = tf_utils.tf_build_top_k_mask_op(input_tensor=y,
                                               k=k_pr,
                                               batch_size=self._hparams.batch_size,
                                               input_area=self._input_size)
        y = y * mask

      # Sum norm (all input is positive)
      # We expect a lot of zeros, or near zeros, and a few larger values.
      # 0, 0.1, 0.1, ... 0.8, 0.9, 0.91
      # 5 / 5 * 20 = 
      # 1/5 = 0.2
      # 5 * 0.2 * 20 = 20
      if self._hparams.cue_nn_sum_norm > 0.0:
        logging.info('PR Sum-norm enabled')
        y_sum = tf.reduce_sum(y, axis=1, keepdims=True)
        reciprocal = 1.0 / y_sum + 0.0000000000001
        y = y * reciprocal * self._hparams.cue_nn_sum_norm

      # Softmax norm
      if self._hparams.cue_nn_softmax is True:
        logging.info('PR Softmax enabled')
        y = tf.nn.softmax(y)
        #y = y * 50.0  # Softmax makes the output *very* small. This hurts Hopfield reconstruction.

      # After norm, 
      if self._hparams.cue_nn_gain != 1.0:
        logging.info('PR Gain enabled')
        y = y * self._hparams.cue_nn_gain

      # Range shift from unit to signed unit
      pr_out = y  # Unit range

      z_cue_in = unit_to_pc_linear(y)  # Theoretical range limits -1 : 1
      target_sparsity = self._hparams.cue_nn_label_sparsity
      shift = get_pc_topk_shift(y, target_sparsity)

      # shift until k bits are > 0, i.e.
      # min *masked* value should become equal to zero.
      z_cue_shift = z_cue_in + shift

      # scaling
      # say scale was y_sum. This is a unit value, so it's always pos.
      # (it sounds like we should have norms on exit of VC etc.)
      # sum =     0.25       0.5
      # -0.5 / 0.25=-2
      #  0.5 / 0.25=2
      # y_sum = tf.reduce_sum(y, axis=1)  # magnitude of output, per batch sample
      # TODO now apply magnitude (y_sum) to the z_cue. just * it?

    else:
      # Old conditioning for PC/Hopfield
      # filter and binarise to get crisp output
      pr_out_mask = tf_utils.tf_build_top_k_mask_op(input_tensor=y,
                                                    k=int(sparsity * pr_sparsity_boost),
                                                    batch_size=self._hparams.batch_size,
                                                    input_area=self._input_size)

      # pr_out = tf.to_float(tf.greater(y, 0.02)) * y    # anything less than 0.2 is 0

      pr_out = pr_out_mask * y                      # use top k mask on y
      z_cue_in = unit_to_pc_sparse(pr_out)          # convert all 0's to -1 for Hopfield
      z_cue_shift = z_cue_in

    self._dual.set_op('pr_out', pr_out)             # raw output of cue_nn
    self._dual.set_op('z_cue_in', z_cue_in)         # modified to pc range and type (binary or real)
    self._dual.set_op('z_cue', z_cue_shift)         # modified to pc range and type (binary or real)
    self._dual.set_op('t_nn', t_nn)                 # Target for NN
    self._dual.set_op('x_nn', x_nn)                 # Target for NN

    # store losses for visualisation
    # ------------------------------------
    # cross entropy loss that is optimized
    loss_pr_memorise = tf.reduce_sum(loss)
    self._dual.set_op('pr_loss', loss_pr_memorise)     # reduced loss

    # number of bits that are mismatched per sample: binary signals, so sum of diff is equal to count of mismatch
    loss_mismatch = tf.reduce_sum(tf.abs(t_nn - pr_out))/self._hparams.batch_size
    self._dual.set_op('pr_loss_mismatch', loss_mismatch)

  def _build_pr_pinv_memorise(self):
    """
    Calculate weights that 'connect' a secondary input to the stored memories
    Use the same approach as pinv.
    (This is used for EC --> CA3 in AMTL project)

    dg = primary (Dentate Gyrus)
    ec = secondary (Entorhinal cortex)
    wp = weights for mapping x_secondary -> hopfield state (perforant)
    """

    x_shape = self._input_values.get_shape().as_list()
    x_size = np.prod(x_shape[1:])

    x = self.get_cue_target()

    self._dual.set_op('pr_target', x)   # in this PR, there is no change between x_ext and pr_target

    x_cue_shape = self._input_cue.get_shape().as_list()
    x_cue_size = np.prod(x_cue_shape[1:])
    x_cue = self._dual.get_op('x_cue')      # EC - secondary, input

    batches = x_shape[0]
    x_matrix = tf.reshape(x, [1, batches, x_size])  # 1 matrix of x vol vecs (expressed as 1 batch)
    x_cue_matrix = tf.reshape(x_cue, [1, batches, x_cue_size])  # 1 matrix of x vol vecs (expressed as 1 batch)
    x_cue_inv = tfp.math.pinv(x_cue_matrix, rcond=None, validate_args=False, name=None)  # this is XT-1 (transposed already)
    w_batches = tf.matmul(x_cue_inv, x_matrix)
    w_p_val = tf.reshape(w_batches, [x_cue_size, x_size])   # strip out the batch dimension

    adjust = 'none'
    # if adjust == 'outgoing':
    #   w_p_exps = tf.exp(w_p_val)
    #   row_sums = tf.reduce_sum(w_p_exps, axis=0, keepdims=True)
    #   w_p_val = tf.divide(w_p_exps, row_sums)
    # elif adjust == 'incoming':
    #   w_p_exps = tf.exp(w_p_val)
    #   col_sums = tf.reduce_sum(w_p_exps, axis=1, keepdims=True)
    #   w_p_val = tf.divide(w_p_exps, col_sums)

    if adjust == 'outgoing':
      # last dimension is columns, or outgoing weights for a given vc neuron
      k_pos = int(self._hparams.cue_nn_label_sparsity * 2)
      k_neg = int(self._hparams.cue_nn_label_sparsity * 10)

      values, indices = tf.nn.top_k(w_p_val, k=k_pos)     # mask for top k
      min_topk = tf.reduce_min(values, axis=1)            # minimum of top k for each column

      values, indices = tf.nn.top_k(-w_p_val, k=k_neg)    # mask for bottom k (topk of negative)
      min_botk = tf.reduce_max(-values, axis=1)           # minimum of top k for each column

      w_p_val = tf.transpose(w_p_val)  # transpose so we can broadcast 'greater' across cols for each row (vc neuron)
      w_mask_topk = tf.to_float(tf.greater_equal(w_p_val, min_topk))      # if it is in the topk, mask it on
      # w_mask_botk = tf.to_float(tf.less_equal(w_p_val, min_botk))     # if it is in the bottk, mask it on

      # # combine masks and apply
      # w_mask = w_mask_topk + w_mask_botk

      # mask for all neg values
      w_mask_neg = tf.to_float(tf.less(w_p_val, 0))
      w_mask = w_mask_topk + w_mask_neg

      w_p_val = tf.multiply(w_mask, w_p_val)
      w_p_val = tf.transpose(w_p_val)  # transpose it back

    w_p_ref = self._dual.get_op('w_p')
    w_p = tf.assign(w_p_ref, w_p_val, name='w_p_assign')
    self._dual.set_op('w_p_assign', w_p)

    z_cue_memorise = tf.matmul(x_cue, w_p)
    pr_loss = tf.reduce_sum(tf.square(x - z_cue_memorise))

    self._dual.set_op('z_cue_memorise', z_cue_memorise)
    self._dual.set_op('pr_loss', pr_loss)

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
      logging.debug('Adam Opt., Hopfield learning rate: ' + str(self._hparams.learning_rate))
      optimizer = tf.train.AdamOptimizer(self._hparams.learning_rate)
    elif self._hparams.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(self._hparams.learning_rate, self._hparams.momentum,
                                             use_nesterov=self._hparams.momentum_nesterov)
    elif self._hparams.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self._hparams.learning_rate)
    else:
      raise NotImplementedError('Optimizer not implemented: ' + str(self._hparams.optimizer))

    return optimizer

  # ---------------- helpers

  @staticmethod
  def _remove_diagonal(tensor):
    mask = np.ones(tensor.get_shape(), dtype=np.float32)
    np.fill_diagonal(mask, 0)
    diagonal_mask = tf.convert_to_tensor(mask)
    weights_updated = tf.multiply(tensor, diagonal_mask)  # must be element-wise
    return weights_updated

  @staticmethod
  def _enforce_symmetry(tensor):
    weights_updated = tf.matrix_band_part(tensor, 0, -1)
    weights_updated = 0.5 * (weights_updated + tf.transpose(weights_updated))
    return weights_updated

  def _is_pinv(self):
    return self._hparams.memorise_method == 'pinv'

  def _is_pinv_hybrid(self):
    return self._hparams.memorise_method == 'pinv_hybrid'

  def use_nn_in_pr_path(self):
    return self._hparams.pr_type == 'nn'

  # ---------------- training

  def update_training_dict(self, feed_dict):
    names = []
    if self.use_inhibition:
      names.extend(['inhibition'])
    self._dual.update_feed_dict(feed_dict, names)

    feed_dict.update({
        self._batch_type: 'training'
    })


  def add_training_fetches(self, fetches):

    names = ['loss_memorise', 'y', 'z_cue', 'pr_out', 'x_direct', 'x_ext']

    if self._is_pinv():
      names.extend(['y_memorise', 'w'])   # need y_memorise to ensure w is assigned

    if self._use_input_cue:
      names.extend(['pr_loss'])
      if self.use_nn_in_pr_path():
        names.extend(['training_pr', 'pr_loss_mismatch'])    # this mismatch loss is more interesting
      else:
        names.extend(['w_p_assign'])

    if self._use_pm:
      names.extend(['training_pm', 'ec_out'])

    if self._use_pm_raw:
      names.extend(['training_pm_raw', 'ec_out_raw'])

    if self.use_inhibition:
      names.extend(['inhibition'])

    # this needs to be done once, because it replaces the fetches, instead of adding to them
    self._dual.add_fetches(fetches, names)

  def set_training_fetches(self, fetched):

    names = ['loss_memorise', 'y', 'z_cue', 'pr_out', 'x_direct', 'x_ext']

    if self._is_pinv():
      names.extend(['w'])

    if self._use_input_cue:
      names.extend(['pr_loss'])   # optional
      if self.use_nn_in_pr_path():
        names.extend(['pr_loss_mismatch'])    # this mismatch loss is more interesting

    if self._use_pm:
      names.extend(['ec_out'])

    if self._use_pm_raw:
      names.extend(['ec_out_raw'])

    if self.use_inhibition:
      names.extend(['inhibition'])

    self._dual.set_fetches(fetched, names)

# ---------------- inference (encoding)

  def update_encoding_dict(self, feed_dict):
    names = []
    if self.use_inhibition:
      names.extend(['inhibition'])
    self._dual.update_feed_dict(feed_dict, names)

    self._update_dict_fb(feed_dict)

    feed_dict.update({
        self._batch_type: 'encoding'
    })

  def add_encoding_fetches(self, fetches):

    names = ['decoding', 'y', 'z_cue', 'pr_out', 'x_direct', 'x_ext']

    if self._is_pinv():
      names.extend(['w'])

    if self._use_input_cue:
      if self.use_nn_in_pr_path():
        names.extend(['pr_loss_mismatch'])    # this mismatch loss is more interesting
        names.extend(['pr_loss'])

    if self._use_pm:
      names.extend(['pm_loss', 'ec_out'])

    if self._use_pm_raw:
      names.extend(['pm_loss_raw', 'ec_out_raw'])

    if self.use_inhibition:
      names.extend(['inhibition'])

    self._dual.add_fetches(fetches, names)

  def set_encoding_fetches(self, fetched):
    names = ['decoding', 'y', 'z_cue', 'pr_out', 'x_direct', 'x_ext']

    if self._is_pinv():
      names.extend(['w'])

    if self._use_input_cue:
      if self.use_nn_in_pr_path():
        names.extend(['pr_loss_mismatch'])    # this mismatch loss is more interesting
        names.extend(['pr_loss'])

    if self._use_pm:
      names.extend(['pm_loss', 'ec_out'])

    if self._use_pm_raw:
      names.extend(['pm_loss_raw', 'ec_out_raw'])

    if self.use_inhibition:
      names.extend(['inhibition'])

    self._dual.set_fetches(fetched, names)

# -------------- build summaries

  def _build_summaries(self, batch_type='training'):
    """Assumes appropriate name scope has been set."""
    summaries = []

    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    if batch_type == 'training':
      summaries = self._build_summaries_memorise(summaries, verbose=False)
    if batch_type == 'encoding':
      summaries = self._build_summaries_retrieve(summaries, verbose=False)
    return summaries

  def _build_recursive_summaries(self):
    """
    Assumes appropriate name scope has been set. Same level as _build_summaries.

    Build same summaries as retrieval.
    """
    summaries = []

    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    summaries = self._build_summaries_retrieve(summaries, verbose=False)
    return summaries

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

      # actual input received to PC from EC (vc out)
      if self.use_pm:
        ec_in = self._input_cue
        ec_out = self._dual.get_op('ec_out')
        ec_recon = image_utils.concat_images([ec_in, ec_out], self._hparams.batch_size)
        summaries.append(tf.summary.image('ec_recon', ec_recon, max_outputs=max_outputs))



        # visualise losses
        pm_loss = self._dual.get_op('pm_loss')
        summaries.append(tf.summary.scalar('pm_loss', pm_loss))

  def _build_summarise_cue_learning(self, summaries, summary_input_shape, batch_type, max_outputs):

    # Terminology:
    # pr = the path to create a cue from VC
    # xcue = the input to the pr path
    # cue = the output of the pr path, used to retrieve a memory from the Hopfield

    if not self._use_input_cue:
      return

    debug_scalar_stats = False

    with tf.name_scope('pr'):

      # nn method
      if self.use_nn_in_pr_path():

        pr_out = self._dual.get_op('pr_out')        # unit
        pr_probs = self._dual.get_op('pr_probs')    # unit
        pr_target = self._dual.get_op('pr_target')  # unit

        # 1) *Mod to input to NN*: show concat of: [x_nn / modified x_nn = x_pr_memorise]
        # modification: may be dropout or noise affected, depending on training/encoding and hparams
        # -------------------------------------------------------------------------------------------
        x_cue = self._dual.get_op('x_cue')
        x_cue_shape = x_cue.get_shape().as_list()
        x_cue_summary_shape, _ = square_image_shape_from_1d(x_cue_shape[1])
        x_cue = tf.reshape(x_cue, x_cue_summary_shape)  # output
        x_pr_memorise = self._dual.get_op('x_pr_memorise')
        x_pr_reshape = tf.reshape(x_pr_memorise, x_cue_summary_shape)  # output
        diff = tf.abs(tf.subtract(x_cue, x_pr_reshape))
        x_cue_mod = image_utils.concat_images([x_cue, x_pr_reshape, diff], self._hparams.batch_size)
        summaries.append(tf.summary.image('xcue_mod', x_cue_mod, max_outputs=max_outputs))

        # 2) *Mod to output of NN* show concat of: [softmax probs / filtered output]
        # -------------------------------------------------------------------------------------------
        prob_out = image_utils.concat_images([pr_probs, pr_out],
                                             self._hparams.batch_size,
                                             summary_input_shape)

        summaries.append(tf.summary.image('prob_out', prob_out, max_outputs=max_outputs))

        # 3) *Results of NN*: show concat of: [labels / filtered output / diff]
        # -------------------------------------------------------------------------------------------
        diff = tf.abs(tf.subtract(pr_target, pr_out))
        concat_image = image_utils.concat_images([pr_target, pr_out, diff], self._hparams.batch_size,
                                                 summary_input_shape)
        summaries.append(tf.summary.image('nn_label_out', concat_image, max_outputs=max_outputs))

        # 4) *Results of pr including space conversions*: show concat of: [raw labels / final output / diff]
        # -------------------------------------------------------------------------------------------
        target = self._dual.get_op('x_ext')
        out = self._dual.get_op('z_cue')
        diff = tf.abs(tf.subtract(target, out))
        concat_image = image_utils.concat_images([target, out, diff], self._hparams.batch_size, summary_input_shape)
        summaries.append(tf.summary.image('pr_tgt_cue', concat_image, max_outputs=max_outputs))

        # 5) *Results of Hopfield given the cue: [labels / internal cue / pc output]
        # -------------------------------------------------------------------------------------------
        y = self._dual.get_op('y')
        concat_image = image_utils.concat_images([pr_target, out, y], self._hparams.batch_size, summary_input_shape)
        summaries.append(tf.summary.image('label_zcue_y', concat_image, max_outputs=max_outputs))

      # pinv method
      else:
        z_cue = self._dual.get_op('z_cue')  # onehot space
        pr_target = self._dual.get_op('pr_target')  # onehot

        show_pr_memorise_with_target = True
        if not show_pr_memorise_with_target:
          summaries.append(tf.summary.image('z_cue', z_cue, max_outputs=max_outputs))
        else:
          label_cue = image_utils.concat_images([pr_target, z_cue], self._hparams.batch_size, summary_input_shape)
          summaries.append(tf.summary.image('label_cue', label_cue, max_outputs=max_outputs))

      # nn method
      if self.use_nn_in_pr_path():
        # visualise losses
        summaries.append(tf.summary.scalar('loss', self._dual.get_op('pr_loss')))

        # count of mismatched bits
        summaries.append(tf.summary.scalar('loss_mismatch', self._dual.get_op('pr_loss_mismatch')))

        summaries.append(tf.summary.histogram('pr_out', self._dual.get_op('pr_out')))

      # pinv method
      else:
        z_cue = self._dual.get_op('z_cue')
        w_p = self._dual.get_op('w_p')

        add_square_as_square(summaries, w_p, 'w_p')

        summaries.append(tf.summary.histogram('w_p', w_p))

        if batch_type == 'training':
          summaries.append(tf.summary.scalar('loss', self._dual.get_op('pr_loss')))

        if z_cue is not None:
          z_cue_reshape = tf.reshape(z_cue, summary_input_shape)
          summaries.append(tf.summary.image('z_cue', z_cue_reshape, max_outputs=max_outputs))

      if debug_scalar_stats:
        summaries.append(tf_utils.tf_build_stats_summaries(self._dual.get_op('x_ext'), 'x_ext'))
        summaries.append(tf_utils.tf_build_stats_summaries(self._dual.get_op('x_cue'), 'x_cue'))
        if self._use_input_cue:
          summaries.append(tf_utils.tf_build_stats_summaries(self._dual.get_op('pr_target'), 'pr_target'))
          summaries.append(tf_utils.tf_build_stats_summaries(self._dual.get_op('pr_out'), 'pr_out'))
          summaries.append(tf_utils.tf_build_stats_summaries(self._dual.get_op('z_cue'), 'z_cue'))

  def _build_summaries_retrieve(self, summaries, verbose=False):
    """Build summaries for retrieval."""

    # summarise_stuff = ['pm', 'pr', 'general']
    # summarise_stuff = ['pm']
    summarise_stuff = ['general']

    max_outputs = self._hparams.max_outputs
    summary_input_shape = image_utils.get_image_summary_shape(self._input_summary_shape)

    if 'general' in summarise_stuff:

      x_direct = self._dual.get_op('x_direct')
      y = self._dual.get_op('y')
      w = self._dual.get_op('w')

      with tf.name_scope('vars'):
        if verbose:
          add_square_as_square(summaries, w, 'w')

          w_p = self._dual.get_op('w_p')
          if w_p is not None:
            add_square_as_square(summaries, w_p, 'w_p')

      if verbose:
        # Inspect data ranges
        ##########################
        x_nn = self._dual.get_op('x_nn')
        t_nn = self._dual.get_op('t_nn')
        p_nn = self._dual.get_op('z_cue_in')  # modified to pc range and type (binary or real)
        l_nn = self._dual.get_op('pr_out')

        summaries.append(tf.summary.histogram('PR_input', x_nn))

        summaries.append(tf.summary.histogram('PR_target', t_nn))
        summaries.append(tf.summary.histogram('PR_predict', p_nn))
        summaries.append(tf.summary.histogram('PR_output', l_nn))
        concat_image = image_utils.concat_images([t_nn, l_nn, y], self._hparams.batch_size, summary_input_shape)
        summaries.append(tf.summary.image('label_zcue_y', concat_image, max_outputs=max_outputs))

      if verbose:
        ops = ['x_ext', 'x_direct', 'z_cue', 'x_fb']
        add_op_images(self._dual, ops, summary_input_shape, max_outputs, summaries)

      x_reshape = tf.reshape(x_direct, summary_input_shape)
      y_reshape = tf.reshape(y, summary_input_shape)

      # output of the net
      summaries.append(tf.summary.image('y', y_reshape, max_outputs=max_outputs))

      show_as_recon_also = True
      if show_as_recon_also:
        diff = tf.abs(tf.subtract(x_reshape, y_reshape))
        x_y = tf.concat([tf.concat([x_reshape, y_reshape], axis=1), diff], axis=1)
        summaries.append(tf.summary.image('x_y_diff', x_y, max_outputs=max_outputs))

      if verbose:
        with tf.name_scope('distr'):

          def add_op_histograms(ops):
            for op_name in ops:
              op = self._dual.get_op(op_name)
              summaries.append(tf.summary.histogram(op_name, op))

          ops = ['w', 'x_direct', 'y']
          add_op_histograms(ops)

      with tf.name_scope('performance'):
        e = self._dual.get_op('e')
        summaries.append(tf.summary.scalar('Energy_total', tf.reduce_sum(e)))

        for idx in range(max_outputs):
          summaries.append(tf.summary.scalar('Energy_'+str(idx), tf.reduce_sum(e[idx])))

    if 'pr' in summarise_stuff:
      self._build_summarise_cue_learning(summaries, summary_input_shape, 'encoding', max_outputs)

    if 'pm' in summarise_stuff:
      self._build_summarise_pm(summaries, max_outputs)

    return summaries

  def _build_summaries_memorise(self, summaries, verbose=False):
    """Build summaries for memorisation."""

    # summarise_stuff = ['pm', 'pr', 'general']
    summarise_stuff = ['pm']
    max_outputs = self._hparams.max_outputs
    summary_input_shape = image_utils.get_image_summary_shape(self._input_summary_shape)

    if 'general' in summarise_stuff:
      w = self._dual.get_op('w')
      k = self._dual.get_op('k')
      b = self._dual.get_op('b')
      loss_memorise = self._dual.get_op('loss_memorise')
      x_ext = self._dual.get_op('x_ext')
      x_direct = self._dual.get_op('x_direct')
      y = self._dual.get_op('y')
      y_memorise = self._dual.get_op('y_memorise')

      ##########################
      # Inspect data ranges
      ##########################
      x_nn = self._dual.get_op('x_nn')
      t_nn = self._dual.get_op('t_nn')
      p_nn = self._dual.get_op('z_cue_in')  # modified to pc range and type (binary or real)
      l_nn = self._dual.get_op('pr_out')
      summaries.append(tf.summary.histogram('PR_input', x_nn))
      summaries.append(tf.summary.histogram('PR_target', t_nn))
      summaries.append(tf.summary.histogram('PR_predict', p_nn))
      summaries.append(tf.summary.histogram('PR_output', l_nn))
      concat_image = image_utils.concat_images([t_nn, l_nn, y], self._hparams.batch_size, summary_input_shape)
      summaries.append(tf.summary.image('label_zcue_y', concat_image, max_outputs=max_outputs))
      ##########################
      # Inspect data ranges
      ##########################

      x_ext_reshaped = tf.reshape(x_ext, summary_input_shape)
      summaries.append(tf.summary.image('x_ext', x_ext_reshaped, max_outputs=max_outputs))

      x_reshape = tf.reshape(x_direct, summary_input_shape)
      summaries.append(tf.summary.image('x_direct', x_reshape, max_outputs=max_outputs))

      if y is not None and not self._is_pinv():
        # doesn't mean anything on pinv, where only one iteration of training, and input was indeterminate
        y_reshape = tf.reshape(y, summary_input_shape)
        summaries.append(tf.summary.image('y', y_reshape, max_outputs=max_outputs))

      if y_memorise is not None:
        y_reshape = tf.reshape(y_memorise, summary_input_shape)
        summaries.append(tf.summary.image('y_memorise', y_reshape, max_outputs=max_outputs))

        show_as_recon_also = True
        if show_as_recon_also:
          diff = tf.abs(tf.subtract(x_reshape, y_reshape))
          x_y = tf.concat([tf.concat([x_reshape, y_reshape], axis=1), diff], axis=1)
          summaries.append(tf.summary.image('x_y_diff', x_y, max_outputs=max_outputs))

      with tf.name_scope('vars'):
        # images
        add_square_as_square(summaries, w, 'w')

        # Monitor parameter sum over time
        with tf.name_scope('sum'):
          w_sum_summary = tf.summary.scalar('w', tf.reduce_sum(tf.abs(w)))
          summaries.extend([w_sum_summary])

        # histograms
        with tf.name_scope('hist'):
          summaries.append(tf.summary.histogram('w', w))

      with tf.name_scope('opt'):

        if loss_memorise is not None:
          summaries.append(tf.summary.scalar('loss_memorise', loss_memorise))

    if 'pr' in summarise_stuff:
      self._build_summarise_cue_learning(summaries, summary_input_shape, 'training', max_outputs)

    if 'pm' in summarise_stuff:
      self._build_summarise_pm(summaries, max_outputs)

    return summaries

  def variables_networks(self, outer_scope):
    vars_nets = []

    # Selectively include/exclude optimizer parameters
    optim_pr = False
    optim_pm = False
    optim_pm_raw = True

    if self._use_input_cue:
      vars_nets += self._variables_cue_nn(outer_scope)
      if optim_pr:
        vars_nets += self._variables_cue_nn_optimizer(outer_scope)

    if self._use_pm:
      vars_nets += self._variables_pm(outer_scope)
      if optim_pm:
        vars_nets += self._variables_pm_optimizer(outer_scope)

    if self.use_pm_raw:
      vars_nets += self._variables_pm_raw(outer_scope)
      if optim_pm_raw:
        vars_nets += self._variables_pm_raw_optimizer(outer_scope)

    return vars_nets

  @staticmethod
  def _variables_cue_nn(outer_scope):
    cue_nn_hidden = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES,
      scope=outer_scope + "/cue_nn_hidden"
    )
    cue_nn_logits = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES,
      scope=outer_scope + "/cue_nn_logits"
    )
    cue_nn = cue_nn_hidden + cue_nn_logits
    return cue_nn

  @staticmethod
  def _variables_pm(outer_scope):
    pm = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES,
      scope=outer_scope + "/pm"
    )
    return pm

  @staticmethod
  def _variables_pm_raw(outer_scope):
    pm_raw = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES,
      scope=outer_scope + "/pm_raw"
    )
    return pm_raw

  @staticmethod
  def _variables_cue_nn_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer/pr")

  @staticmethod
  def _variables_pm_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer/pm")

  @staticmethod
  def _variables_pm_raw_optimizer(outer_scope):
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=outer_scope + "/optimizer/pm_raw")
