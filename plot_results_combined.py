"""Parse AHA results from Jenkins output."""

import os
import re
import argparse

from collections import OrderedDict, namedtuple

import numpy as np

import matplotlib.pyplot as plt


from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', './builds', 'Path to directory containing logfiles.')
flags.DEFINE_integer('num_seeds', 10, 'Number of seeds.', lower_bound=1)
flags.DEFINE_enum('exp', 'oneshot', ['oneshot', 'instance'], 'Experiment type.')
flags.DEFINE_enum('metric', 'class', ['class', 'replay'], 'Metric for the plots.')
flags.DEFINE_enum('perturb', 'occ', ['occ', 'noise'], 'Perturbation type.')
flags.DEFINE_list('models', 'aha,ae', 'Models to include in plots.')

def load_and_parse_data(filepath):
  """Load and parse the output from experiment sweeps."""
  with open(filepath, 'r') as f:
    data = f.readlines()

  key = 'acc_mse_vc,'   # keyword to find the heading line
  line_offset = 2       # how many lines apart are the headings and values

  c = 0
  value_idxs = []
  heading_idxs = []

  for num, line in enumerate(data):
    if key in line:
      heading_idxs.append(num)

    if c < len(heading_idxs) and num == heading_idxs[c] + line_offset:
      value_idxs.append(num)
      c += 1

  def parse_csv(array, idxs):
    return [re.sub(r'\s+', '', array[i]).strip().split(',')[:-1] for i in idxs]

  values = parse_csv(data, value_idxs)
  headings = parse_csv(data, heading_idxs) # headings are identical, only need one copy

  # Convert values from strings => floats
  for i, row in enumerate(values):
    for j, item in enumerate(row):
      values[i][j] = float(item)

  return headings, values


def concatenate_results(all_headings, all_values):
  """Concatenate results from multiple runs and group by the key e.g. acc_vc = [0.5, 0.6, ...]"""
  num_sweeps = len(all_values)
  num_values = len(all_values[0])

  results = OrderedDict()
  for k in all_headings[0][0]:
    results[k] = np.zeros([num_sweeps, num_values])

  for i, (sweep_headings, sweep_values) in enumerate(zip(all_headings, all_values)):

    for j, (headings, values) in enumerate(zip(sweep_headings, sweep_values)):
      for k, v in zip(headings, values):
        results[k][i][j] = v

  return results

def compute_statistics(results, thresh=None):
  """Compute summary statistics (min, max, mean, std, etc.) for each key in results."""
  results_stats = OrderedDict()
  summary_stats = namedtuple('summary_stats', 'mean, se, sd, count, mins, maxs')

  def reject_outliers(data, m=1):
    mask = abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0)
    masked_array = np.ma.masked_array(data=data, mask=~mask)

    # fill_value = np.max(masked_array)
    fill_value = np.mean(masked_array)
    # fill_value = 0.0

    return masked_array.filled(fill_value)

  for k, v in results.items():
    if thresh is not None:
      if k == 'acc_mse_pm_raw':
        new_v = reject_outliers(v)
        for i, _ in enumerate(v):
          print('before', v[i], '\n')
          print('after', new_v[i], '\n\n')
        v = new_v

    count = len(v)
    se = np.std(v, axis=0) / count
    sd = (se * count).mean()

    results_stats[k] = summary_stats(mins=np.min(v, axis=0),
                                     maxs=np.max(v, axis=0),
                                     mean=np.mean(v, axis=0),
                                     se=se,
                                     sd=sd,
                                     count=count)

  return results_stats


def build_xaxis(num, radius_increment=0.05):
  """Calculate the radius/diameter for the x-axis in plots."""
  x_axis = {
      'radius': [],
      'diameter': []
  }

  radius = 0
  for _ in range(num):
    radius = round(radius, 2)
    diameter = round(radius * 2, 1)
    x_axis['radius'].append(radius)
    x_axis['diameter'].append(diameter)
    radius += radius_increment

  return x_axis


def get_filenames(dirpath):
  filenames = []
  for root, _, files in os.walk(dirpath):
    for file in files:
      if file.endswith('.log') or file.endswith('.txt'):
        filepath = os.path.join(root, file)
        filenames.append(filepath)
  return filenames

def plot_mean_sd(ax, xaxis, vals, ses, sd, label, color, mins, maxs, dashes=(None, None), alpha=0.08,
                 with_range=False):
  """Plot with optional error shadows."""
  # print(label, "-> SD = %.3f," % sd, "Best mean = %.3f," % vals.max(), "Best max = %.3f" % maxs.max())

  LW = 0.7
  ax.plot(xaxis, vals, label=label, c=color, dashes=dashes, linewidth=LW)
  ax.fill_between(xaxis, vals-ses, vals+ses, alpha=alpha, color=color)
  if with_range:
    ax.fill_between(xaxis, mins, maxs, alpha=alpha/2, color=color)


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


def main(_):
  exp = FLAGS.exp
  metric = FLAGS.metric
  perturb = FLAGS.perturb
  model_names = FLAGS.models
  num_seeds = FLAGS.num_seeds
  input_path = FLAGS.input_path

  print('Models =', model_names)
  print('Mode =', exp, metric, perturb, '\n')

  models = {}

  for model in model_names:
    filename = model + '-' + exp + '-' + 'class' + '-' + perturb + '.log'
    filepath = os.path.join(input_path, filename)

    headings, values = load_and_parse_data(filepath)

    all_headings = list(chunks(headings, num_seeds + 1))
    all_values = list(chunks(values, num_seeds + 1))

    num_items = len(all_values[0])

    models[model] = {}
    models[model]['results'] = concatenate_results(all_headings, all_values)
    models[model]['results_stats'] = compute_statistics(models[model]['results'])

  xaxes = build_xaxis(num_items)
  xaxis = xaxes['diameter']

  _, ax = plt.subplots(1, 1, dpi=250, figsize=(10, 5))

  if exp == 'oneshot' and perturb == 'occ':
    title = 'One-shot classification with occlusion'
  elif exp == 'oneshot' and perturb == 'noise':
    title = 'One-shot classification with noise'
  elif exp == 'instance' and perturb == 'occ':
    title = 'One-shot instance-classification with occlusion'
  elif exp == 'instance' and perturb == 'noise':
    title = 'One-shot instance-classification with noise'

  if metric == 'class':
    ylabel = 'Accuracy'
    ymax = 1.0
  elif metric == 'replay':
    ylabel = 'Recall Loss'
    # ymax = 0.3
    ymax = None

  if metric == 'class':
    if 'aha' in models:
      vc_key = 'acc_mse_vc'
      plot_mean_sd(ax, xaxis,
                   vals=models['aha']['results_stats'][vc_key].mean,
                   ses=models['aha']['results_stats'][vc_key].se,
                   sd=models['aha']['results_stats'][vc_key].sd,
                   mins=models['aha']['results_stats'][vc_key].mins,
                   maxs=models['aha']['results_stats'][vc_key].maxs,
                   label='LTM',
                   color='blue',
                   dashes=(None, None),
                   with_range=True,
                   alpha=0.1)

      pc_key = 'acc_mse_pc'
      plot_mean_sd(ax, xaxis,
                   vals=models['aha']['results_stats'][pc_key].mean,
                   ses=models['aha']['results_stats'][pc_key].se,
                   sd=models['aha']['results_stats'][pc_key].sd,
                   mins=models['aha']['results_stats'][pc_key].mins,
                   maxs=models['aha']['results_stats'][pc_key].maxs,
                   label='LTM+AHA-PC',
                   color='red',
                   dashes=(6, 1),
                   with_range=True,
                   alpha=0.1)

      pr_key = 'acc_mse_pc_in'
      plot_mean_sd(ax, xaxis,
                   vals=models['aha']['results_stats'][pr_key].mean,
                   ses=models['aha']['results_stats'][pr_key].se,
                   sd=models['aha']['results_stats'][pr_key].sd,
                   mins=models['aha']['results_stats'][pr_key].mins,
                   maxs=models['aha']['results_stats'][pr_key].maxs,
                   label='LTM+AHA-PR',
                   color='orange',
                   dashes=(2, 1),
                   with_range=True,
                   alpha=0.1)

      print('LTM Accuracy =',
            models['aha']['results_stats'][vc_key].mean[0],
            models['aha']['results_stats'][vc_key].se[0])

      print('LTM+AHA-PC Accuracy =',
            models['aha']['results_stats'][pc_key].mean[0],
            models['aha']['results_stats'][pc_key].se[0])

      print('LTM+AHA-PR Accuracy =',
            models['aha']['results_stats'][pr_key].mean[0],
            models['aha']['results_stats'][pr_key].se[0])

      print('\n')

    if 'ae' in models:
      ae_key = 'acc_mse_pc'
      plot_mean_sd(ax, xaxis,
                   vals=models['ae']['results_stats'][ae_key].mean,
                   ses=models['ae']['results_stats'][ae_key].se,
                   sd=models['ae']['results_stats'][ae_key].sd, label='LTM+FastNN',
                   mins=models['ae']['results_stats'][ae_key].mins,
                   maxs=models['ae']['results_stats'][ae_key].maxs,
                   color='green',
                   dashes=(2, 1),
                   with_range=True,
                   alpha=0.1)

      print('LTM+FastNN Accuracy =',
            models['ae']['results_stats'][ae_key].mean[0],
            models['ae']['results_stats'][ae_key].se[0])

    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')

    if ymax is not None:
      ax.set_ylim((0, ymax))

  elif metric == 'replay':
    replay_key = 'acc_mse_pm_raw'
    ae_color = 'green'
    aha_color = 'red'

    single_plot = True

    if 'aha' in models:
      plot_mean_sd(ax, xaxis,
                   vals=models['aha']['results_stats'][replay_key].mean,
                   ses=models['aha']['results_stats'][replay_key].se,
                   sd=models['aha']['results_stats'][replay_key].sd,
                   mins=models['aha']['results_stats'][replay_key].mins,
                   maxs=models['aha']['results_stats'][replay_key].maxs,
                   label='LTM+AHA',
                   color=aha_color,
                   dashes=(None, None),
                   with_range=True,
                   alpha=0.1)

      print('LTM+AHA Recall Loss =',
            models['aha']['results_stats'][replay_key].mean[0],
            models['aha']['results_stats'][replay_key].se[0])

    if not single_plot:
      ax.set_ylabel(ylabel, color=aha_color)
      ax.tick_params(axis='y', labelcolor=aha_color)
      ax.legend(loc='upper left')

      ax2 = ax.twinx()
    else:
      ax2 = ax

    if 'ae' in models:
      plot_mean_sd(ax2, xaxis,
                   vals=models['ae']['results_stats'][replay_key].mean,
                   ses=models['ae']['results_stats'][replay_key].se,
                   sd=models['ae']['results_stats'][replay_key].sd, label='LTM+FastNN',
                   mins=models['ae']['results_stats'][replay_key].mins,
                   maxs=models['ae']['results_stats'][replay_key].maxs,
                   color=ae_color,
                   dashes=(2, 1),
                   with_range=(None, None),
                   alpha=0.1)

      print('LTM+FastNN Recall Loss =',
            models['ae']['results_stats'][replay_key].mean[0],
            models['ae']['results_stats'][replay_key].se[0])

    if not single_plot:
      ax2.set_ylabel(ylabel, color=ae_color)
      ax2.tick_params(axis='y', labelcolor=ae_color)
      ax2.legend(loc='upper right')
    else:
      ax.set_ylabel(ylabel)
      ax.legend(loc='upper right')

  ax.set_title(title)

  if perturb == 'noise':
    ax.set_xlabel('Proportion')
  elif perturb == 'occ':
    ax.set_xlabel('Diameter')

  ax.set_xlim((0, max(xaxis)))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Pick top accuracy without occlusion/noise as ceiling
  if metric == 'class':
    acc_ceil = 0.0

    if 'aha' in models:
      acc_ceil = max(models['aha']['results_stats'][vc_key].mean[0],
                     models['aha']['results_stats'][pc_key].mean[0],
                     models['aha']['results_stats'][pr_key].mean[0])

    if 'ae' in models:
      acc_ceil = max(acc_ceil, models['ae']['results_stats'][ae_key].mean[0])

    ax.plot(ax.get_xlim(), [acc_ceil, acc_ceil], c='gray', dashes=[4, 2], linewidth=0.9)

  filename = exp + '_' + metric + '_' + perturb + '.png'
  plt.savefig(filename)
  # plt.show()

if __name__ == '__main__':
  app.run(main)
