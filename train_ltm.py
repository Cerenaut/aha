"""Train LTM on 20 Omniglot runs."""

import os
import datetime
import subprocess

NUM_RUNS = 20
DEFINITION_PATH = 'definitions/epw-omni-pretrain-ll.json'


def main():
  experiment_prefix = datetime.datetime.now().strftime('%y%m%d-%H%M')

  for r in range(1, NUM_RUNS + 1):
    run_folder = 'run' + str(r).zfill(2)
    workflow_opts_sweep = {
        'evaluate_mode': ['simple', run_folder]
    }

    now = datetime.datetime.now()
    summary_dir = 'summaries_' + now.strftime("%Y%m%d-%H%M%S") + '/'
    summary_path = os.path.join('run', experiment_prefix, summary_dir)

    subprocess.call([
        'pagi',
        'run',
        '--experiment_def=' + DEFINITION_PATH,
        '--workflow_opts_sweep=' + str(workflow_opts_sweep),
        '--summary_dir=' + summary_path
    ])

if __name__ == '__main__':
  main()
