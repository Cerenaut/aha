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

"""EpisodicComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import sys
import logging

import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils, generic_utils
from pagi.utils.dual import DualData
from pagi.utils.hparam_multi import HParamMulti
from pagi.utils.layer_utils import type_activation_fn
from pagi.utils.np_utils import np_uniform
from pagi.utils.tf_utils import tf_build_interpolate_distributions

from pagi.components.summarize_levels import SummarizeLevels
from pagi.components.composite_component import CompositeComponent
from pagi.components.visual_cortex_component import VisualCortexComponent
from pagi.components.sparse_autoencoder_component import SparseAutoencoderComponent
from pagi.components.sparse_conv_maxpool import SparseConvAutoencoderMaxPoolComponent

from aha.components.dg_sae import DGSAE
from aha.components.dg_scae import DGSCAE
from aha.components.dg_stub import DGStubComponent
from aha.components.label_learner_fc import LabelLearnerFC
from aha.components.hopfieldlike_component import HopfieldlikeComponent
from aha.components.deep_autoencoder_component import DeepAutoencoderComponent
from aha.components.diff_plasticity_component import DifferentiablePlasticityComponent

from aha.utils.interest_filter import InterestFilter
from aha.utils.generic_utils import normalize_minmax


class PCMode(enum.Enum):
  PCOnly = 1   # Use only PC
  Exclude = 2  # Use all sub-components excluding PC i.e. everything up to PC
  Combined = 3  # Use all sub-components including PC


HVC_ENABLED = True  # Hierarchical VC


class EpisodicComponent(CompositeComponent):
  """
  A component to implement episodic memory, inspired by the Medial Temporal Lobe.
  Currently, it consists of a Visual Cortex (Sparse Autoencoder, SAE) and
  a Pattern Completer similar to DG/CA3 (Differentiable Plasticity or SAE).
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""

    # create component level hparams (this will be a multi hparam, with hparams from sub components)
    batch_size = 40
    max_outputs = 3
    hparam = tf.contrib.training.HParams(
        batch_size=batch_size,
        output_features='pc',  # the output of this subcomponent is used as the component's features
        pc_type='sae',         # none, hl = hopfield like, sae = sparse autoencoder, dp = differentiable-plasticity
        dg_type='fc',          # 'none', 'fc', or 'conv' Dentate Gyrus
        ll_vc_type='none',     # vc label learner: 'none', 'fc'
        ll_pc_type='none',     # pc label learner: 'none', 'fc'
        use_cue_to_pc=False,   # use a secondary input as a cue to pc (EC perforant path to CA3)
        use_pm=False,          # pattern mapping (reconstruct inputs from PC output
        use_interest_filter=False,  # this replaces VC (attentional system zones in on interesting features)
        summarize_level=SummarizeLevels.ALL.value,  # for the top summaries (leave individual comps to decide on own)
        vc_norm_per_filter=False,
        vc_norm_per_sample=False,
        max_pool_vc_final_size=2,
        max_pool_vc_final_stride=1,
        max_outputs=max_outputs
    )

    # create all possible sub component hparams (must create one for every possible sub component)
    if HVC_ENABLED:
      vc = VisualCortexComponent.default_hparams()
    else:
      vc = SparseConvAutoencoderMaxPoolComponent.default_hparams()

    dg_fc = DGSAE.default_hparams()
    dg_conv = DGSCAE.default_hparams()
    dg_stub = DGStubComponent.default_hparams()
    pc_sae = SparseAutoencoderComponent.default_hparams()
    pc_dae = DeepAutoencoderComponent.default_hparams()
    pc_dp = DifferentiablePlasticityComponent.default_hparams()
    pc_hl = HopfieldlikeComponent.default_hparams()
    ifi = InterestFilter.default_hparams()
    ll_vc = LabelLearnerFC.default_hparams()
    ll_pc = LabelLearnerFC.default_hparams()

    subcomponents = [vc, dg_fc, dg_conv, dg_stub, pc_sae, pc_dae, pc_dp, pc_hl, ll_vc, ll_pc]   # all possible subcomponents

    # default overrides of sub-component hparam defaults
    if not HVC_ENABLED:
      vc.set_hparam('learning_rate', 0.001)
      vc.set_hparam('sparsity', 25)
      vc.set_hparam('sparsity_output_factor', 1.5)
      vc.set_hparam('filters', 64)
      vc.set_hparam('filters_field_width', 6)
      vc.set_hparam('filters_field_height', 6)
      vc.set_hparam('filters_field_stride', 3)

      vc.set_hparam('pool_size', 2)
      vc.set_hparam('pool_strides', 2)

      # Note that DG will get the pooled->unpooled encoding
      vc.set_hparam('use_max_pool', 'none')  # none, encoding, training

    dg_fc.set_hparam('learning_rate', 0.001)
    dg_fc.set_hparam('sparsity', 20)
    dg_fc.set_hparam('sparsity_output_factor', 1.0)
    dg_fc.set_hparam('filters', 784)

    pc_hl.set_hparam('learning_rate', 0.0001)
    pc_hl.set_hparam('optimizer', 'adam')
    pc_hl.set_hparam('momentum', 0.9)
    pc_hl.set_hparam('momentum_nesterov', False)
    pc_hl.set_hparam('use_feedback', True)
    pc_hl.set_hparam('memorise_method', 'pinv')
    pc_hl.set_hparam('nonlinearity', 'none')
    pc_hl.set_hparam('update_n_neurons', -1)

    # default hparams in individual component should be consistent with component level hparams
    HParamMulti.set_hparam_in_subcomponents(subcomponents, 'batch_size', batch_size)

    # add sub components to the composite hparams
    HParamMulti.add(source=vc, multi=hparam, component='vc')
    HParamMulti.add(source=dg_fc, multi=hparam, component='dg_fc')
    HParamMulti.add(source=dg_conv, multi=hparam, component='dg_conv')
    HParamMulti.add(source=dg_stub, multi=hparam, component='dg_stub')
    HParamMulti.add(source=pc_dp, multi=hparam, component='pc_dp')
    HParamMulti.add(source=pc_sae, multi=hparam, component='pc_sae')
    HParamMulti.add(source=pc_dae, multi=hparam, component='pc_dae')
    HParamMulti.add(source=pc_hl, multi=hparam, component='pc_hl')
    HParamMulti.add(source=ifi, multi=hparam, component='ifi')
    HParamMulti.add(source=ll_vc, multi=hparam, component='ll_vc')
    HParamMulti.add(source=ll_pc, multi=hparam, component='ll_pc')

    return hparam

  def __init__(self):
    super(EpisodicComponent, self).__init__()

    self._name = None
    self._hparams = None
    self._summary_op = None
    self._summary_result = None
    self._dual = None
    self._input_shape = None
    self._input_values = None
    self._summary_values = None

    self._sub_components = {}  # map {name, component}

    self._pc_mode = PCMode.Combined

    self._pc_input = None
    self._pc_input_vis_shape = None

    self._degrade_type = 'random'  # if degrading is used, then a degrade type: vertical, horizontal, random

    self._signals = {}  # signals at each stage: convenient container for significant signals
    self._show_episodic_level_summary = True

    self._interest_filter = None

  def batch_size(self):
    return self._hparams.batch_size

  def get_vc_encoding(self):
    return self._dual.get_values('vc_encoding')

  def is_build_dg(self):
    return self._hparams.dg_type != 'none'

  def is_build_ll_vc(self):
    return self._hparams.ll_vc_type != 'none'

  def is_build_ll_pc(self):
    return self._hparams.ll_pc_type != 'none'

  def is_build_ll_ensemble(self):
    build_ll_ensemble = True
    return self.is_build_ll_vc() and self.is_build_ll_pc() and build_ll_ensemble

  def is_build_pc(self):
    return self._hparams.pc_type != 'none'

  def is_pc_hopfield(self):
    return isinstance(self.get_pc(), HopfieldlikeComponent)

  @staticmethod
  def is_vc_hierarchical():
    # return isinstance(self.get_vc(), VisualCortexComponent)
    return HVC_ENABLED    # need to use this before _component is instantiated

  def pc_combined(self):
    self._pc_mode = PCMode.Combined

  def pc_exclude(self):
    self._pc_mode = PCMode.Exclude

  def pc_only(self):
    self._pc_mode = PCMode.PCOnly

  @property
  def name(self):
    return self._name

  def get_interest_filter_masked_encodings(self):
    return self._dual.get_values('masked_encodings')

  def get_interest_filter_positional_encodings(self):
    return self._dual.get_values('positional_encodings')

  def set_signal(self, key, val, val_shape):
    """Set as a significant signal, that should be summarised"""
    self._signals.update({key: (val, val_shape)})

  def get_signal(self, key):
    val, val_shape = self._signals[key]
    return val, val_shape

  def get_loss(self):
    """Define loss as the loss of the subcomponent selected for output features: using _hparam.output_features"""

    if self._hparams.output_features == 'vc':
      comp = self.get_vc()
    elif self._hparams.output_features == 'dg':
      comp = self.get_dg()
    else:  # assumes     output_features == 'pc'
      comp = self.get_pc()
    return comp.get_loss()

  @staticmethod
  def degrader(degrade_step_pl, degrade_type, random_value_pl, input_values, degrade_step, name=None):

    return tf.cond(tf.equal(degrade_step_pl, degrade_step),
                   lambda: image_utils.degrade_image(input_values,
                                                     degrade_type=degrade_type,
                                                     random_value=random_value_pl),
                   lambda: input_values,
                   name=name)

  def _build_vc(self, input_values, input_shape):

    if HVC_ENABLED:
      vc = VisualCortexComponent()
      hparams_vc = VisualCortexComponent.default_hparams()
    else:
      vc = SparseConvAutoencoderMaxPoolComponent()
      hparams_vc = SparseConvAutoencoderMaxPoolComponent.default_hparams()

    hparams_vc = HParamMulti.override(multi=self._hparams, target=hparams_vc, component='vc')
    vc.build(input_values, input_shape, hparams_vc, 'vc')
    self._add_sub_component(vc, 'vc')

    # Update 'next' value/shape for DG
    if HVC_ENABLED:
      # Since pooling/unpooling is applied within the VC component,
      # use the get_output() method to get the final layer of VC with the
      # appropriate pooling/unpooling setting.
      input_values_next = vc.get_output_op()
    else:
      # Otherwise, get the encoding or unpooled encoding as appropriate
      input_values_next = vc.get_encoding_op()
      if hparams_vc.use_max_pool == 'encoding':
        input_values_next = vc.get_encoding_unpooled_op()
    print('vc', 'output', input_values_next)

    # Optionally norm VC per filter (this should probably be only first layer, but only one layer for now anyway)
    for _, layer in vc.get_sub_components().items():
      layer.set_norm_filters(self._hparams.vc_norm_per_filter)

    # Add InterestFilter to mask in and blur position of interesting visual filters (and block out the rest)
    if self._hparams.use_interest_filter:
      self._interest_filter = InterestFilter()
      image_tensor, image_shape = self.get_signal('input')
      vc_tensor = input_values_next
      vc_shape = vc_tensor.get_shape().as_list()

      assert image_shape[:-1] == vc_shape[:-1], "The VC encoding must be the same height and width as the image " \
        "i.e. conv stride 1"

      hparams_ifi = InterestFilter.default_hparams()
      hparams_ifi = HParamMulti.override(multi=self._hparams, target=hparams_ifi, component='ifi')
      _, input_values_next = self._interest_filter.build(image_tensor, vc_tensor, hparams_ifi)
      self._dual.set_op('masked_encodings', self._interest_filter.get_image('masked_encodings'))
      self._dual.set_op('positional_encodings', self._interest_filter.get_image('positional_encodings'))

    # Optionally pool the final output of the VC (simply to reduce dimensionality)
    pool_size = self._hparams.max_pool_vc_final_size
    pool_stride = self._hparams.max_pool_vc_final_stride
    if pool_size > 1:
      input_values_next = tf.layers.max_pooling2d(input_values_next, pool_size, pool_stride, 'SAME')
      print('vc final pooled', input_values_next)

    # Optionally norm the output samples so that they are comparable to the next stage
    def normalize_min_max_4d(x):
      sample_mins = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
      sample_maxs = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
      return (x - sample_mins) / (sample_maxs - sample_mins)

    if self._hparams.vc_norm_per_sample:
      frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(input_values_next), axis=[1, 2, 3], keepdims=True))
      input_values_next = input_values_next / frobenius_norm
      #input_values_next = normalize_min_max_4d(input_values_next)

    # Unpack the conv cells shape
    input_volume = np.prod(input_values_next.get_shape().as_list()[1:])
    input_next_vis_shape, _ = image_utils.square_image_shape_from_1d(input_volume)

    return input_values_next, input_next_vis_shape

  def _build_ll_vc(self, target_output, train_input, test_input, name='ll_vc'):
    """Build the label learning component for LTM."""
    ll_vc = None

    # Don't normalize this yet
    train_input = normalize_minmax(train_input)
    test_input = normalize_minmax(test_input)

    if self._hparams.ll_vc_type == 'fc':
      ll_vc = LabelLearnerFC()
      self._add_sub_component(ll_vc, name)
      hparams_ll_vc = LabelLearnerFC.default_hparams()
      hparams_ll_vc = HParamMulti.override(multi=self._hparams, target=hparams_ll_vc, component='ll_vc')
      ll_vc.build(target_output, train_input, test_input, hparams_ll_vc, name)

    return ll_vc

  def _build_ll_pc(self, target_output, train_input, test_input, name='ll_pc'):
    """Build the label learning component for PC."""
    ll_pc = None

    train_input = normalize_minmax(train_input)
    test_input = normalize_minmax(test_input)

    if self._hparams.ll_pc_type == 'fc':
      ll_pc = LabelLearnerFC()
      self._add_sub_component(ll_pc, name)
      hparams_ll_pc = LabelLearnerFC.default_hparams()
      hparams_ll_pc = HParamMulti.override(multi=self._hparams, target=hparams_ll_pc, component='ll_pc')
      ll_pc.build(target_output, train_input, test_input, hparams_ll_pc, name)

    return ll_pc

  def _build_dg(self, input_next, input_next_vis_shape):
    """Builds the pattern separation component."""
    dg_type = self._hparams.dg_type

    if dg_type == 'stub':

      # create fc, so that we can use the encodings etc. without breaking other stuff
      dg = DGStubComponent()
      self._add_sub_component(dg, 'dg')
      hparams_dg = DGStubComponent.default_hparams()
      hparams_dg = HParamMulti.override(multi=self._hparams, target=hparams_dg, component='dg_stub')

      dg.build(hparams_dg, 'dg')

      # Update 'next' value/shape for PC
      input_next = dg.get_encoding_op()
      input_next_vis_shape, _ = image_utils.square_image_shape_from_1d(hparams_dg.filters)

      dg_sparsity = hparams_dg.sparsity
    elif dg_type == 'fc':
      dg = DGSAE()
      self._add_sub_component(dg, 'dg')
      hparams_dg = DGSAE.default_hparams()
      hparams_dg = HParamMulti.override(multi=self._hparams, target=hparams_dg, component='dg_fc')
      dg.build(input_next, input_next_vis_shape, hparams_dg, 'dg')

      # Update 'next' value/shape for PC
      input_next = dg.get_encoding_op()
      input_next_vis_shape, _ = image_utils.square_image_shape_from_1d(hparams_dg.filters)

      dg_sparsity = hparams_dg.sparsity
    elif dg_type == 'conv':
      input_next_vis_shape = [-1] + input_next.get_shape().as_list()[1:]

      print('dg', 'input', input_next)

      dg = DGSCAE()
      self._add_sub_component(dg, 'dg')
      hparams_dg = DGSCAE.default_hparams()
      hparams_dg = HParamMulti.override(multi=self._hparams, target=hparams_dg, component='dg_conv')
      dg.build(input_next, input_next_vis_shape, hparams_dg, 'dg')

      # Update 'next' value/shape for PC
      input_next = dg.get_encoding_op()

      print('dg', 'output', input_next)

      # Unpack the conv cells shape
      input_volume = np.prod(input_next.get_shape().as_list()[1:])
      input_next_vis_shape, _ = image_utils.square_image_shape_from_1d(input_volume)
      input_next = tf.reshape(input_next, [-1, input_volume])

      dg_sparsity = hparams_dg.sparsity

    else:
      raise NotImplementedError('Dentate Gyrus not implemented: ' + dg_type)

    return input_next, input_next_vis_shape, dg_sparsity

  def _build_pc(self, input_next, input_next_vis_shape, dg_sparsity):

    pc_type = self._hparams.pc_type
    use_cue_to_pc = self._hparams.use_cue_to_pc
    use_pm = self._hparams.use_pm

    if use_cue_to_pc:
      cue = self._signals['vc'][0]
    else:
      cue = None

    cue_raw = None
    if use_pm:
      cue_raw = self._signals['vc_input'][0]

    if pc_type == 'sae':
      pc = SparseAutoencoderComponent()
      self._add_sub_component(pc, 'pc')
      hparams_sae = SparseAutoencoderComponent.default_hparams()
      hparams_sae = HParamMulti.override(multi=self._hparams, target=hparams_sae, component='pc_sae')
      pc.build(input_next, input_next_vis_shape, hparams_sae, 'pc')
    elif pc_type == 'dae':
      pc = DeepAutoencoderComponent()
      self._add_sub_component(pc, 'pc')
      hparams_dae = DeepAutoencoderComponent.default_hparams()
      hparams_dae = HParamMulti.override(multi=self._hparams, target=hparams_dae, component='pc_dae')
      input_next_shape = input_next.get_shape().as_list()
      pc.build(input_next, input_next_shape, hparams_dae, 'pc', input_cue_raw=cue_raw)
    elif pc_type == 'dp':
      # DP works with batches differently, so not prescriptive for input shape (used for summaries only)
      input_next_vis_shape[0] = -1

      # ensure DP receives binary values (all k winners will be 1)
      input_next = tf.greater(input_next, 0)
      input_next = tf.to_float(input_next)

      pc = DifferentiablePlasticityComponent()
      self._add_sub_component(pc, 'pc')
      hparams_pc_dp = DifferentiablePlasticityComponent.default_hparams()
      hparams_pc_dp = HParamMulti.override(multi=self._hparams, target=hparams_pc_dp, component='pc_dp')
      pc.build(input_next, input_next_vis_shape, hparams_pc_dp, 'pc')
    elif pc_type == 'hl':
      pc = HopfieldlikeComponent()
      self._add_sub_component(pc, 'pc')
      hparams_hl = HopfieldlikeComponent.default_hparams()
      hparams_hl = HParamMulti.override(multi=self._hparams, target=hparams_hl, component='pc_hl')

      if dg_sparsity == 0:
        raise RuntimeError("Could not establish dg per sample sparsity to pass to Hopfield.")

      hparams_hl.cue_nn_label_sparsity = dg_sparsity

      pc.build(input_next, input_next_vis_shape, hparams_hl, 'pc', input_cue=cue, input_cue_raw=cue_raw)
    else:
      raise NotImplementedError('Pattern completer not implemented: ' + pc_type)

    pc_output = pc.get_decoding_op()

    if pc_type == 'dae':
      input_volume = np.prod(pc_output.get_shape().as_list()[1:])
      pc_output_shape, _ = image_utils.square_image_shape_from_1d(input_volume)
    else:
      pc_output_shape = input_next_vis_shape  # output is same shape and size as input

    return pc_output, pc_output_shape

  def _build_ll_ensemble(self):
    """Builds ensemble of VC and PC classifiers."""
    distributions = []
    distribution_mass = []
    num_classes = self._label_values.get_shape().as_list()[-1]

    aha_mass = 0.495
    ltm_mass = 0.495
    uniform_mass = 0.01

    if aha_mass > 0.0:
      aha_prediction = self.get_ll_pc().get_op('preds')
      distributions.append(aha_prediction)
      distribution_mass.append(aha_mass)

    if ltm_mass > 0.0:
      ltm_prediction = self.get_ll_vc().get_op('preds')
      distributions.append(ltm_prediction)
      distribution_mass.append(ltm_mass)

    if uniform_mass > 0.0:
      uniform = np_uniform(num_classes)
      distributions.append(uniform)
      distribution_mass.append(uniform_mass)

    unseen_sum = 1
    unseen_idxs = (0, unseen_sum)

    # Build the final distribution, calculate loss
    ensemble_preds = tf_build_interpolate_distributions(distributions, distribution_mass, num_classes)

    ensemble_correct_preds = tf.equal(tf.argmax(ensemble_preds, 1), tf.argmax(self._label_values, 1))
    ensemble_correct_preds = tf.cast(ensemble_correct_preds, tf.float32)

    ensemble_accuracy = tf.reduce_mean(ensemble_correct_preds)
    ensemble_accuracy_unseen = tf.reduce_mean(ensemble_correct_preds[unseen_idxs[0]:unseen_idxs[1]])

    self._dual.set_op('ensemble_preds', ensemble_preds)
    self._dual.set_op('ensemble_accuracy', ensemble_accuracy)
    self._dual.set_op('ensemble_accuracy_unseen', ensemble_accuracy_unseen)

  def build(self, input_values, input_shape, hparams, label_values=None, name='episodic'):
    """Initializes the model parameters.

    Args:
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        :param input_values:
        :param input_shape:
        :param hparams:
        :param name:
    """

    self._name = name
    self._hparams = hparams
    self._summary_op = None
    self._summary_result = None
    self._dual = DualData(self._name)
    self._input_values = input_values
    self._input_shape = input_shape
    self._label_values = label_values

    input_area = np.prod(input_shape[1:])

    logging.debug('Input Shape: %s', input_shape)
    logging.debug('Input Area: %s', input_area)

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

      # Replay mode
      # ------------------------------------------------------------------------
      replay_mode = 'pixel'   # pixel or encoding
      replay = self._dual.add('replay', shape=[], default_value=False).add_pl(
          default=True, dtype=tf.bool)

      # Replace labels during replay
      replay_labels = self._dual.add('replay_labels', shape=label_values.shape, default_value=0.0).add_pl(
          default=True, dtype=label_values.dtype)

      self._label_values = tf.cond(tf.equal(replay, True), lambda: replay_labels, lambda: self._label_values)

      # Replay pixel inputs during replay, if using 'pixel' replay mode
      if replay_mode == 'pixel':
        replay_inputs = self._dual.add('replay_inputs', shape=input_values.shape, default_value=0.0).add_pl(
            default=True, dtype=input_values.dtype)

        self._input_values = tf.cond(tf.equal(replay, True), lambda: replay_inputs, lambda: self._input_values)

      self.set_signal('input', self._input_values, self._input_shape)

      # Build the LTM
      # ------------------------------------------------------------------------

      # Optionally degrade input to VC
      degrade_step_pl = self._dual.add('degrade_step', shape=[],  # e.g. hidden, input, none
                                       default_value='none').add_pl(default=True, dtype=tf.string)
      degrade_random_pl = self._dual.add('degrade_random', shape=[],
                                         default_value=0.0).add_pl(default=True, dtype=tf.float32)
      input_values = self.degrader(degrade_step_pl, self._degrade_type, degrade_random_pl, self._input_values,
                                   degrade_step='input', name='vc_input_values')

      print('vc', 'input', input_values)
      self.set_signal('vc_input', input_values, input_shape)

      # Build the VC
      input_next, input_next_vis_shape = self._build_vc(input_values, input_shape)

      vc_encoding = input_next

      # Replace the encoding during replay, if using 'encoding' replay mode
      if replay_mode == 'encoding':
        replay_inputs = self._dual.add('replay_inputs', shape=vc_encoding.shape, default_value=0.0).add_pl(
            default=True, dtype=vc_encoding.dtype)

        vc_encoding = tf.cond(tf.equal(replay, True), lambda: replay_inputs, lambda: vc_encoding)

      self.set_signal('vc', vc_encoding, input_next_vis_shape)
      self._dual.set_op('vc_encoding', vc_encoding)

      # Build the softmax classifier
      if self.is_build_ll_vc() and self._label_values is not None:
        self._build_ll_vc(self._label_values, vc_encoding, vc_encoding)

      # Build AHA
      # ------------------------------------------------------------------------

      # Build the DG
      dg_sparsity = 0
      if self.is_build_dg():
        input_next, input_next_vis_shape, dg_sparsity = self._build_dg(input_next, input_next_vis_shape)
        dg_encoding = input_next
        self.set_signal('dg', dg_encoding, input_next_vis_shape)

      # Build the PC
      if self.is_build_pc():
        # Optionally degrade input to PC

        # not all degrade types are supported for embedding in graph (but may still be used directly on test set)
        if self._degrade_type != 'rect' and self._degrade_type != 'circle':
          input_next = self.degrader(degrade_step_pl, self._degrade_type, degrade_random_pl, input_next,
                                     degrade_step='hidden', name='pc_input_values')
        print('pc_input', input_next)
        self.set_signal('pc_input', input_next, input_next_vis_shape)

        pc_output, pc_output_shape = self._build_pc(input_next, input_next_vis_shape, dg_sparsity)
        self.set_signal('pc', pc_output, pc_output_shape)

        if self._hparams.use_pm:
          ec_out_raw = self.get_pc().get_ec_out_raw_op()
          self.set_signal('ec_out_raw', ec_out_raw, input_shape)

        if self.is_build_ll_pc() and self.is_build_dg() and self._label_values is not None:
          self._build_ll_pc(self._label_values, dg_encoding, pc_output)

      if self.is_build_ll_ensemble():
        self._build_ll_ensemble()

    self.reset()

  def get_vc(self):
    vc = self.get_sub_component('vc')
    return vc

  def get_pc(self):
    pc = self.get_sub_component('pc')
    return pc

  def get_dg(self):
    dg = self.get_sub_component('dg')
    return dg

  def get_decoding(self):
    return self.get_pc().get_decoding()

  def get_ll_vc(self):
    return self.get_sub_component('ll_vc')

  def get_ll_pc(self):
    return self.get_sub_component('ll_pc')

  def get_batch_type(self, name=None):
    """
    Return dic of batch types for each component (key is component)
    If component does not have a persistent batch type, then don't include in dictionary,
    assumption is that in that case, it doesn't have any effect.
    """
    if name is None:
      batch_types = dict.fromkeys(self._sub_components.keys(), None)
      for c in self._sub_components:
        if hasattr(self._sub_components[c], 'get_batch_type'):
          batch_types[c] = self._sub_components[c].get_batch_type()
        else:
          batch_types.pop(c)
      return batch_types

    return self._sub_components[name].get_batch_type()

  def get_features(self, batch_type='training'):
    """
    The output of the component is taken from one of the subcomponents, depending on hparams.
    If not vc or dg, the fallback is to take from pc regardless of value of the hparam
    """
    del batch_type
    if self._hparams.output_features == 'vc':
      features = self.get_vc().get_features()
    elif self._hparams.output_features == 'dg':
      features = self.get_dg().get_features()
    else:  # self._hparams.output_features == 'pc':
      features = self.get_pc().get_features()
    return features

  def _is_skip_for_pc(self, name):
    if self._pc_mode == PCMode.PCOnly:  # only pc
      if name != 'pc':
        return True
    elif self._pc_mode == PCMode.Exclude:  # skip pc
      if name == 'pc':
        return True
    return False

  def update_feed_dict_input_gain_pl(self, feed_dict, gain):
    """
    This is relevant for the PC, and is only be called when it is being run recursively.
    """
    if self.get_pc() is not None:
      self.get_pc().update_feed_dict_input_gain_pl(feed_dict, gain)

  def update_feed_dict(self, feed_dict, batch_type='training'):
    for name, comp in self._sub_components.items():
      if self._is_skip_for_pc(name):
        continue
      comp.update_feed_dict(feed_dict, self._select_batch_type(batch_type, name))

  def add_fetches(self, fetches, batch_type='training'):
    # each component adds its own
    for name, comp in self._sub_components.items():
      if self._is_skip_for_pc(name):
        continue
      comp.add_fetches(fetches, self._select_batch_type(batch_type, name))

    # Episodic Component specific
    # ------------------------------
    # Interest Filter and other
    names = []

    if self._hparams.use_interest_filter:
      names.extend(['masked_encodings', 'positional_encodings'])

    if self.is_build_ll_ensemble():
      names.extend(['ensemble_preds', 'ensemble_accuracy', 'ensemble_accuracy_unseen'])

    # Other
    names.extend(['vc_encoding'])

    if len(names) > 0:
      self._dual.add_fetches(fetches, names)

    # Episodic Component specific - summaries
    bt = self._select_batch_type(batch_type, self._name)
    summary_op = self._dual.get_op(generic_utils.summary_name(bt))
    if summary_op is not None:
      fetches[self._name]['summaries'] = summary_op

  def set_fetches(self, fetched, batch_type='training'):
    # each component adds its own
    for name, comp in self._sub_components.items():
      if self._is_skip_for_pc(name):
        continue
      comp.set_fetches(fetched, self._select_batch_type(batch_type, name))

    # Episodic Component specific
    # ----------------------------
    # Interest Filter
    names = []

    if self._hparams.use_interest_filter:
      names.extend(['masked_encodings', 'positional_encodings'])

    if self.is_build_ll_ensemble():
      names.extend(['ensemble_preds', 'ensemble_accuracy', 'ensemble_accuracy_unseen'])

    # other
    names.extend(['vc_encoding'])

    if len(names) > 0:
      self._dual.set_fetches(fetched, names)

    # Episodic Component specific - summaries
    bt = self._select_batch_type(batch_type, self._name)
    summary_op = self._dual.get_op(generic_utils.summary_name(bt))
    if summary_op is not None:
      self._summary_values = fetched[self._name]['summaries']

  def build_summaries(self, batch_types=None):
    if batch_types is None:
      batch_types = []

    components = self._sub_components.copy()

    consolidate_graph_view = False

    if self._show_episodic_level_summary:
      components.update({self._name: self})

    for name, comp in components.items():
      scope = name + '-summaries'   # this is best for visualising images in summaries
      if consolidate_graph_view:
        scope = self._name + '/' + name + '/summaries/'

      bt = self._select_batch_type(batch_types, name, as_list=True)

      if name == self._name:
        comp.build_summaries_episodic(bt, scope=scope)
      else:
        comp.build_summaries(bt, scope=scope)

  def write_summaries(self, step, writer, batch_type='training'):
    # the episodic component itself
    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)    # Write the summaries fetched into _summary_values
      writer.flush()

    super().write_summaries(step, writer, batch_type)

  def write_recursive_summaries(self, step, writer, batch_type=None):
    for name, comp in self._sub_components.items():
      if hasattr(comp, 'write_recursive_summaries'):
        comp.write_recursive_summaries(step, writer, batch_type)

  def build_summaries_episodic(self, batch_types=None, scope=None):
    """Builds all summaries."""

    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:

        # build 'batch_type' summary subgraph
        with tf.name_scope(batch_type):
          summaries = self._build_summaries(batch_type)
          if len(summaries) > 0:
            self._dual.set_op(generic_utils.summary_name(batch_type), tf.summary.merge(summaries))

  def _build_summaries(self, batch_type='training'):
    """Assumes appropriate name scope has been set."""
    max_outputs = self._hparams.max_outputs
    summaries = []

    if self._hparams.summarize_level != SummarizeLevels.OFF.value:
      for key, pair in self._signals.items():
        val = pair[0]
        val_shape = pair[1]

        summary_shape = image_utils.get_image_summary_shape(val_shape)
        reshaped = tf.reshape(val, summary_shape)
        summaries.append(tf.summary.image(key, reshaped, max_outputs=max_outputs))

    if self._hparams.use_interest_filter and self._interest_filter.summarize_level() != SummarizeLevels.OFF.value:
      with tf.name_scope('interest_filter'):
        self._interest_filter.add_summaries(summaries)

    return summaries
