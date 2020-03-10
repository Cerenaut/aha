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
File containing utils that could be utilities for other workflows (but have not been refactored completely as yet).
They have also been moved here to make few_shot workflow much cleaner and easy to work with.
"""

import os

import numpy as np

from sklearn import metrics
from scipy.spatial.distance import cdist

from pagi.utils import np_utils, image_utils
from aha.utils.generic_utils import overlap_match


def mod_hausdorff_distance(x, y):
    # Modified Hausdorff Distance
  #
  # Input
  #  x : [n x 2] coordinates of "inked" pixels
  #  y : [m x 2] coordinates of "inked" pixels
  #
  #  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
  #  International Conference on Pattern Recognition, pp. 566-568.
  #
  dist = cdist(x, y)
  mindist_x = dist.min(axis=1)
  mindist_y = dist.min(axis=0)
  mean_x = np.mean(mindist_x)
  mean_y = np.mean(mindist_y)
  return max(mean_x, mean_y)


def add_completion_summary(summary_images, folder, summary, batch, save_figs):
  """
  Show all the relevant images put in _summary_images by the _prep methods.
  They are collected during training and testing.

  NOTE: summary_images is a LIST and is plotted in subfigure in the given order

  """

  render_rows = False       # in Tensorboard
  render_subplots = True    # with Matplotlib

  plot_encoding = True
  plot_diff = True

  if len(summary_images) == 3:  #  3 images -> train_input, test_input, recon (i.e. no encoding)
    plot_encoding = False

  # in Tensorboard
  if render_rows:
    for idx, (name, image, image_shape) in enumerate(summary_images):
      # print(name, image_shape)
      image = np.reshape(image, image_shape)  # reshape in case we want to render individually

      # stick the batch samples together to make one long image - a row of images
      row_shape = [1, image_shape[1], image_shape[0] * image_shape[2], image_shape[3]]
      row = np.reshape(np.transpose(image, axes=(1, 0, 2, 3)), row_shape)
      image_utils.arbitrary_image_summary(summary, row, name='pcw/' + name, max_outputs=3)
      np_utils.np_write_array_list_as_image(folder, row, str(idx) + '_' + name)

  # in Matplotlib
  if render_subplots:
    import matplotlib.pyplot as plt
    if save_figs:
      plt.switch_backend('agg')

    col_nums = True
    row_nums = True

    # first_image
    (name, image, image_shape) = summary_images[0]

    rows = len(summary_images)
    rows = rows + (2 if plot_diff else 0)
    rows = rows + (1 if col_nums else 0)
    cols = image_shape[0]  + 1 if row_nums else 0  # number of samples in batch

    if plot_encoding:
      # figsize = [18, 5.4]  # figure size, inches
      figsize = [10, 3]  # figure size, inches
    else:
      figsize = [10, 2]

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, num=batch)
    plt.subplots_adjust(left=0, right=1.0, bottom=0, top=1.0, hspace=0.1, wspace=0.1)

    # plot simple raster image on each sub-plot
    for i, ax in enumerate(ax.flat):
      # i runs from 0 to (nrows*ncols-1)
      # axi is equivalent with ax[rowid][colid]

      # get indices of row/column
      row_idx = i // cols
      col_idx = i % cols

      if col_idx == 0:
        if row_idx == rows - 1:
          ax.text(0.3, 0.3, ' ')
        else:
          ax.text(0.3, 0.3, str(row_idx+1))
      elif plot_diff and row_idx in [rows - 2, rows - 3]:
        if col_idx == 0:
          ax.text(0.3, 0.3, ' ')
        else:
          img_idx = col_idx - 1

          _, target_imgs, target_shape = summary_images[0]
          target_shape = [target_shape[1], target_shape[2]]
          _, output_imgs, _ = summary_images[-1]

          target_img = np.reshape(target_imgs[img_idx], target_shape)
          output_img = np.reshape(output_imgs[img_idx], target_shape)
          output_img = np.clip(output_img, 0.0, 1.0)

          sq_err = np.square(target_img - output_img)

          mse = sq_err.mean()
          mse = '{:.2E}'.format(mse)

          if row_idx == rows - 2:
            ax.text(0.3, 0.3, mse, fontsize=5)
          elif row_idx == rows - 3:
            ax.imshow(sq_err)

      elif row_idx == rows - 1:
        if col_idx == 0:
          ax.text(0.3, 0.3, ' ')
        else:
          ax.text(0.3, 0.3, str(col_idx))
      else:
        (name, image, image_shape) = summary_images[row_idx]
        image_shape = [image_shape[1], image_shape[2]]

        img_idx = col_idx - 1
        img = np.reshape(image[img_idx], image_shape)

        if not plot_encoding or name in ['vc_input', 'ec_out_raw']:
          ax.imshow(img, cmap='binary', vmin=0, vmax=1)
        else:
          ax.imshow(img, vmin=-1, vmax=1)

      ax.axis('off')

    if save_figs:
      filetype = 'png'
      filename = 'completion_summary_' + str(batch) + '.' + filetype
      filepath = os.path.join(folder, filename)
      plt.savefig(filepath, dpi=300, format=filetype)
    else:
      plt.show()


def add_completion_summary_paper(summary_images, folder, batch, save_figs):
  """
  Show all the relevant images put in _summary_images by the _prep methods.
  They are collected during training and testing.

  NOTE: summary_images is a LIST and is plotted in subfigure in the given order

  """

  import matplotlib.pyplot as plt
  if save_figs:
    plt.switch_backend('agg')

  # first_image
  (name, image, image_shape) = summary_images[0]

  rows = len(summary_images)
  cols = image_shape[0]   # number of samples in batch

  # figsize = [11, 1]       # figure size, inches

  # plot simple raster image on each sub-plot
  for r in range(rows):

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=1, ncols=cols, num=batch)    # figsize=figsize,
    # plt.subplots_adjust(left=0.5, hspace=0.25, wspace=0.25)

    for i, ax in enumerate(ax.flat):

      (name, image, image_shape) = summary_images[r]
      image_shape = [image_shape[1], image_shape[2]]

      img = np.reshape(image[i], image_shape)

      if r == 0 or r == 2 or r == 5:
        ax.imshow(img, cmap='binary', vmin=0, vmax=1)
      else:
        ax.imshow(img, vmin=-1, vmax=1)

      # write row/col indices as axes' title for identification
      # axi.set_title(name + str(col_idx))
      ax.axis('off')

    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)

    if save_figs:
      filetype = 'png'
      filename = 'completion_summary_{}_{}.{}'.format(batch, r, filetype)
      filepath = os.path.join(folder, filename)
      plt.savefig(filepath, dpi=300, format=filetype)
    else:
      plt.show()

  # if save_figs:
  #   filetype = 'png'
  #   filename = 'completion_summary_' + str(batch) + '.' + filetype
  #   filepath = os.path.join(folder, filename)
  #   plt.savefig(filepath, dpi=300, format=filetype)
  # else:
  #   plt.show()


def create_and_add_comparison_image(summary_images, batch_size, name, train_key, test_key, use_mse=True, k=-1,
                                    train_labels=None, test_labels=None):
  """
  Create image for each test sample, showing similarity between all train samples
  More similar = closer to 1, or brighter.

  Use mse if the negative space is also important, as for Hopfield in (-1, +1) mode.

  Add the image to summary_images, which is then used to create a Matplotlib figure.
  """

  # compare retrieved vc_encodings (after convergence) with all the vc_encodings in the memorisation (train) set
  (train_name, train_image, train_image_shape) = summary_images[train_key]
  (test_name,   test_image, test_image_shape) = summary_images[test_key]

  strip_width = 4
  row_pixels = batch_size * strip_width
  col_height_pixels = row_pixels // 2
  diffs = np.zeros(
    [batch_size, col_height_pixels, row_pixels])  # [batch_size, num comparisons with train batch=batch_size]

  render_truth_image = False
  if train_labels is not None and test_labels is not None:
    render_truth_image = True
    col_height_pixels_half = int(np.math.ceil(0.5 * col_height_pixels))

  max_vals = -float('inf') * np.ones(batch_size)
  min_vals = float('inf') * np.ones(batch_size)
  min_j = -1 * np.ones(batch_size)
  max_j = -1 * np.ones(batch_size)
  for i, image_i in enumerate(train_image[:]):  # batch samples
    for j, image_j in enumerate(test_image[:]):  # cols
      x = j * strip_width
      if use_mse:
        val = np.square(image_i - image_j).mean()
      else:
        val = overlap_match(image_i, image_j)

      diffs[i, :, x:x + strip_width - 1] = val

      if val > max_vals[i]:
        max_vals[i] = val
        max_j[i] = j

      if val < min_vals[i]:
        min_vals[i] = val
        min_j[i] = j

  if render_truth_image:
    for i, image_i in enumerate(train_image[:]):  # batch samples
      for j, image_j in enumerate(test_image[:]):  # cols
        x = j * strip_width

        h2 = int(col_height_pixels_half / 2)

        # show best match by extending that bar lower than the others
        if use_mse and j == min_j[i]:
          val = min_vals[i]
        elif not use_mse and j == max_j[i]:
          val = max_vals[i]
        else:
          val = -1.0

        diffs[i, col_height_pixels_half:col_height_pixels_half+h2, x:x + strip_width - 1] = val

        # show truth
        if test_labels[i] == train_labels[j]:
          val = 1
        else:
          val = -1

        diffs[i, col_height_pixels_half+h2:col_height_pixels, x:x + strip_width - 1] = val

  # calc accuracy for display
  truth_indices = np.arange(batch_size)
  if use_mse:
    predictions = min_j
  else:
    predictions = max_j

  accuracy = metrics.accuracy_score(truth_indices, predictions)
  print("ComparisonImage - Accuracy: {0}".format(accuracy))

  # normalise values so range is (-1, +1), for display
  max_all = np.max(max_vals)
  if k == -1:
    k = 1 / max_all
  diffs[:, 0:col_height_pixels_half, :] *= k

  # brighter is 'more overlap' using mse, so need to invert (mse is difference, the opposite)
  if use_mse:
    diffs[:, 0:col_height_pixels_half, :] = 1 - diffs[:, 0:col_height_pixels_half, :]

  im_name = "{0}\n{1}\n(1/{2:.5f})".format(name, accuracy, max_all)
  summary_images[name] = (im_name, diffs, [-1, col_height_pixels, row_pixels, 1])


def compute_matching(modes, train_features, test_features, comparison_type='mse'):
  """
  For each of the possible metrics (feature types), calculate and return a similarity matrix and scalar accuracy.

  @:param train_features = dic(feature_type, [label 1 value, label 2 value, label 3 value, ......])
  feature_type e.g. 'dg_hidden', 'dg_recon'
    special case is `labels`
  @:param test_features same format as train_features
  @:param mode = 'oneshot' or 'instance' (see compute_truth_matrix_[oneshot/instance] for description)
  @:param comparison_type_ type of comparison to use, could be 'mse' or 'overlap'
  @:param test_first compares all train sample to a given test sample (instead of the opposite, which is the default)
  @:return matching_matrices, matching_accuracies: {feature_type --> matrix/accuracies}
  """

  def compute_matrix_prep(feature_type):
    if feature_type not in train_features.keys() or feature_type not in test_features.keys():
      return None

    train_ftrs = train_features[feature_type]  # shape = [num labels, feature_size]
    test_ftrs = test_features[feature_type]  # shape = [num labels, feature_size]

    num_labels = train_ftrs.shape[0]
    matrix = np.zeros([num_labels, num_labels])

    return train_ftrs, test_ftrs, num_labels, matrix

  def compute_matrix(feature_type, comparison_type_='mse'):
    """
    Compute a 'confusion matrix' style matrix of the mse between test and train sets - for a specified feature type
    For feature_type='labels', compute float equivalent of True(1.0)/False(0.0) if they are the same or not.

    @:param feature_type the name (key in dic train_features and test_features) of features to use

                         Test Label
    Train Label |    Label 1              Label 2              Label 3
    ----------------------------------------------------------------------------------
    Label 1     |     mse(trn_1, tst_1)   mse(trn_1, tst_2)   mse(trn_1, tst_2)
    Label 2     |     mse(trn_2, tst_1)   ...                  ...
    Label 3     |     mse(trn_3, tst_1)   ...                  ...
    Label 4     |     mse(trn_4, tst_1)   ...                  ...

    """

    train_ftrs, test_ftrs, num_labels, matrix = compute_matrix_prep(feature_type)

    for i in range(num_labels):
      train = train_ftrs[i]
      for j in range(num_labels):
        test = test_ftrs[j]

        if comparison_type_ == 'mse':
          matrix[i, j] = np.square(train - test).mean()
        else:   # overlap
          matrix[i, j] = overlap_match(train, test)

    return matrix

  def compute_truth_matrix_oneshot():
    """
    Compute truth matrix for oneshot test
    Find matching labels in Test and Train and set that cell 'true match'
    """

    train_ftrs, test_ftrs, num_labels, matrix = compute_matrix_prep('labels')

    for idx_train in range(num_labels):
      for idx_test in range(num_labels):
        if train_ftrs[idx_train] == test_ftrs[idx_test]:
          matrix[idx_train, idx_test] = 1.0
        else:
          matrix[idx_train, idx_test] = 0.0
    return matrix

  def compute_truth_matrix_instance():
    """
    Compute truth matrix for test of instance learning
    Test and Train are the same exemplars, and we are matching to exact exemplar, so diagonals are 'true match'.
    """

    # the feature_type passed to prep is irrelevant, we just want data structures it creates
    _, _, num_labels, matrix = compute_matrix_prep('labels')

    for idx_train in range(num_labels):
      for idx_test in range(num_labels):
        if idx_train == idx_test:  # the same exemplar, so 'correct' answer
          matrix[idx_train, idx_test] = 1.0
        else:
          matrix[idx_train, idx_test] = 0.0
    return matrix

  def compute_accuracy(similarity, comparison_type_='mse'):
    """

    @:param similarity: matrix [train x test], size=[num labels x num labels], elements=similarity score
    @:param comparison_type_: function type used for comparisons, could be 'mse' or 'overlap'
    @:return:
    """

    dbug = False
    if dbug:
      print("------------ COMPUTE ACCURACY ---------------")
      predictions = np.argmin(similarity, axis=1)    # argmin for each train sample, rows (across test samples, cols)
      truth = np.argmax(truth_matrix, axis=1)
      acc = metrics.accuracy_score(truth, predictions)
      print(similarity)
      print(truth_matrix)
      print(predictions)
      print(truth)
      print("Accuracy = {0}".format(acc))

    num_labels = similarity.shape[0]
    correct = 0
    max_correct = 0
    sum_ambiguous_ = 0
    for i in range(num_labels):

      if comparison_type_ == 'mse':
        best_test_idx = np.argmin(similarity[i, :])  # this constitutes the prediction
      else:   # overlap
        best_test_idx = np.argmax(similarity[i, :])  # this constitutes the prediction

      best_val = similarity[i, best_test_idx]
      bast_val_indices = np.where(similarity[i] == best_val)
      if len(bast_val_indices[0]) > 1:
        sum_ambiguous_ += 1

      num_correct = np.sum(truth_matrix[i, :])  # is there a correct answer (i.e. was there a matching class?)
      if num_correct > 0:
        max_correct = max_correct + 1

      val = truth_matrix[i, best_test_idx]  # was this one of the matching classes (there may be more than 1)

      if val == 1.0:
        correct = correct + 1

    if max_correct == 0:
      accuracy = -1.0
    else:
      accuracy = correct / max_correct
    return accuracy, sum_ambiguous_

  if 'oneshot' in modes:
    truth_matrix = compute_truth_matrix_oneshot()
  else:
    truth_matrix = compute_truth_matrix_instance()

  matching_matrices = {'train_labels': train_features['labels'],
                       'test_labels': test_features['labels'],
                       'truth': truth_matrix}
  matching_accuracies = {}
  sum_ambiguous = {}
  for feature_type in test_features.keys():
    if feature_type == 'labels':
      continue

    if feature_type not in train_features.keys():
      continue

    feature_matrix = compute_matrix(feature_type, comparison_type)
    matching_matrices[feature_type] = feature_matrix
    matching_accuracies[feature_type], sum_ambiguous[feature_type] = compute_accuracy(feature_matrix, comparison_type)

  return matching_matrices, matching_accuracies, sum_ambiguous
