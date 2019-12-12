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
Episodic-specific general utils.
"""

import numpy as np
import tensorflow as tf

def normalize_minmax(inputs):
  return (inputs - tf.reduce_min(inputs)) / (tf.reduce_max(inputs) - tf.reduce_min(inputs))


def print_minmax(tensor, name):
  return tf.Print(tensor, [tf.reduce_min(tensor), tf.reduce_max(tensor), tf.reduce_sum(tensor)], str(name) + ' - min/max/sum = ')


def build_kernel_initializer(init_type='he', uniform=False):
  if init_type == 'he':
    init_factor = 2.0
    init_mode = 'FAN_IN'
  elif init_type == 'xavier':
    init_factor = 1.0
    init_mode = 'FAN_AVG'
  else:
    raise NotImplementedError('Initializer not supported: ' + str(init_type))
  return tf.contrib.layers.variance_scaling_initializer(factor=init_factor, mode=init_mode, uniform=uniform)


def overlap(a, b):
  """ a, b = (0,1)
  Overlap is where tensors have 1's in the same position.
  Return number of bits that overlap """
  return np.sum(np.abs(a*b))


def overlap_match(a, b):
  """ a, b = (0,1)
  Overlap is where tensors have 1's in the same position.
  Return number of bits that overlap """
  return np.sum(a*b)


def compute_overlap(results, vals, labels, vals_test, labels_test, per_label=False, scope=""):
  """
  Computes a number of overlap metrics between two batches of samples in params (vals, labels)
  i.e. compares every pair in vals with vals_test
  if vals_test or labels_test == None, then compare all pairs within the batch:

  Put the results in the 'results' dictionary, with the name of keys prepended with scope

  - intra and inter: overlap matrices between samples in the set
                   intra - overlap between same samples of the same class
                   inter - overlap between samples of different classes
  - count_intra, count_inter: the number of intra and inter comparisons that were made
              (the batch is a random sample, so this is to check there were enough, mainly for intra)
  - inter_per_label: same as above, but produce a matrix for each label separately
  - error rate: frequency of max inter_overlap > max intra_overlap per sample
              i.e. 0.5 means that 1/2 samples have bad encoding"""

  same_batch = False
  if vals_test is None or labels_test is None:
    vals_test = vals
    labels_test = labels
    same_batch = True

  n = len(vals)
  intra = []
  inter = []
  sample_labels = []
  sample_preds = []

  inter_per_label = {}
  intra_per_label = {}

  # list of diff between max inter and intra. If it is correct (intra>inter) how much is it correct by?
  # If it is wrong, how much is it wrong by?
  accuracy_margin = []

  if per_label:
    for label in labels:
      inter_per_label[label] = []
      intra_per_label[label] = []

  error_count = 0
  for i, (val_i, label_i) in enumerate(zip(vals, labels)):

    # keep track, per batch, of overlaps, to count errors
    max_overlap_label = -1
    max_overlap = -float('inf')
    max_intra_overlap = max_overlap
    max_inter_overlap = max_overlap

    for j, (val_j, label_j) in enumerate(zip(vals_test, labels_test)):

      if same_batch and i == j:
        continue  # ignore self

      olap = overlap(val_i, val_j)

      if olap > max_overlap:
        max_overlap = olap
        max_overlap_label = label_j

      # ---- intra
      if label_j == label_i:
        intra.append(olap)
        if olap > max_intra_overlap:
          max_intra_overlap = olap
        if per_label:
          intra_per_label[label_i].append(olap)

      # ---- inter
      else:  # Dissimilar labels
        inter.append(olap)
        if olap > max_inter_overlap:
          max_inter_overlap = olap
        if per_label:
          inter_per_label[label_i].append(olap)

    sample_labels.append(label_i)           # Record true label
    sample_preds.append(max_overlap_label)  # Record predicted label

    # Determine if sample i could possibly be classified correctly.
    accuracy_margin.append(max_intra_overlap - max_inter_overlap)   # +ve = correct
    if max_inter_overlap > max_intra_overlap:
      error_count = error_count + 1

  error_rate = error_count / n

  # "a confusion matrix C is such that C_ij is equal to the number of observations known to be in group i
  # but predicted to be in group j."
  cm = metrics.confusion_matrix(sample_labels, sample_preds, sample_labels)
  acc = metrics.accuracy_score(sample_labels, sample_preds)

  # set results
  # ----------------------------------------

  if scope != "":
    scope = scope + "_"

  results[scope + 'accuracy'] = acc
  results[scope + 'confusion'] = cm
  results[scope + 'accuracy_margin'] = accuracy_margin

  if per_label:
    results[scope + 'inter_per_label'] = inter_per_label
    results[scope + 'intra_per_label'] = intra_per_label

  results[scope + 'error_rate'] = error_rate

  results[scope + 'inter'] = inter
  results[scope + 'intra'] = intra

  # results[scope + '_count_inter'] = len(inter)
  # results[scope + '_count_intra'] = len(intra)


def overlap_sample_batch(i, vals, labels, vals_test, labels_test):
  """
  Calculate intra and inter overlap for sample at index `i` with the rest of the batch given by `batch_data`
  """

  same_batch = False
  if vals_test is None or labels_test is None:
    vals_test = vals
    labels_test = labels
    same_batch = True

  (val_i, label_i) = (vals[i], labels[i])

  intra = []
  inter = []
  for j, (val_j, label_j) in enumerate(zip(vals_test, labels_test)):
    if same_batch and i == j:
      continue  # ignore self
    olap = overlap(val_i, val_j)

    if label_j == label_i:
      intra.append(olap)
    else:
      inter.append(olap)

  return inter, intra
