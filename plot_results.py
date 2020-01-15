"""Parse AHA results from Jenkins output."""

import os
import re
import argparse

from collections import OrderedDict, namedtuple

import numpy as np

import matplotlib.pyplot as plt


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

def compute_statistics(results):
  """Compute summary statistics (min, max, mean, std, etc.) for each key in results."""
  results_stats = OrderedDict()
  summary_stats = namedtuple('summary_stats', 'mean, se, sd, count, mins, maxs')

  for k, v in results.items():
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


def plot_mean_sd(ax, xaxis, vals, ses, sd, label, color, mins, maxs, dashes=(None, None), alpha=0.08,
                 with_range=False):
  """Plot with optional error shadows."""
  LW = 0.7
  ax.plot(xaxis, vals, label=label, c=color, dashes=dashes, linewidth=LW)
  ax.fill_between(xaxis, vals-ses, vals+ses, alpha=alpha, color=color)
  if with_range:
    ax.fill_between(xaxis, mins, maxs, alpha=alpha/2, color=color)
  # print(label, "SD %.3f" % sd, "Best mean LSTM %.3f" % vals.max(), "best max %.3f" % maxs.max(), "mean %% of ceil %.3f" % (vals.max() / ACC_CEIL))


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--input_path', type=str, help='an integer for the accumulator')

  args = parser.parse_args()

  if args.input_path is None or args.input_path == '' or not os.path.exists(args.input_path):
    raise ValueError('Input path does not exist.')

  filenames = []
  for root, dirs, files in os.walk(args.input_path):
    for file in files:
      if file.endswith('.log'):
        filepath = os.path.join(root, file)
        filenames.append(filepath)

  all_headings, all_values = [], []
  for filepath in filenames:
    headings, values = load_and_parse_data(filepath)
    all_headings.append(headings)
    all_values.append(values)

  num_items = len(all_values[0])
  # num_values = len(max(values, key=len))

  results = concatenate_results(all_headings, all_values)
  results_stats = compute_statistics(results)

  xaxes = build_xaxis(num_items)
  xaxis = xaxes['diameter']

  fig, ax = plt.subplots(1, 1, dpi=250, figsize=(8, 4))

  vc_key = 'acc_mse_vc'
  plot_mean_sd(ax, xaxis, results_stats[vc_key].mean, results_stats[vc_key].se, results_stats[vc_key].sd, 'VC', 'blue',
               results_stats[vc_key].mins, results_stats[vc_key].maxs, (None, None), with_range=True, alpha=0.1)

  pc_key = 'acc_mse_pc'
  plot_mean_sd(ax, xaxis, results_stats[pc_key].mean, results_stats[pc_key].se, results_stats[pc_key].sd, 'PC', 'red',
               results_stats[pc_key].mins, results_stats[pc_key].maxs, (6, 1), with_range=True, alpha=0.1)

  pr_key = 'acc_mse_pc_in'
  plot_mean_sd(ax, xaxis, results_stats[pr_key].mean, results_stats[pr_key].se, results_stats[pr_key].sd, 'PR', 'orange',
               results_stats[pr_key].mins, results_stats[pr_key].maxs, (2, 1), with_range=True, alpha=0.1)

  acc_ceil = results_stats[pr_key].mean[0]  # Pick accuracy without occlusion/noise as ceiling

  ax.set_title('Classification with occlusion')
  ax.set_xlabel('Diameter')
  ax.legend(loc='upper right')
  ax.set_ylabel('Accuracy')
  ax.set_ylim((0, 1.0))
  ax.set_xlim((0, max(xaxis)))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.plot(ax.get_xlim(), [acc_ceil, acc_ceil], c='gray', dashes=[4, 2], linewidth=0.9)

  plt.savefig('classification_occlusion.png')
  plt.show()

if __name__ == '__main__':
  main()
