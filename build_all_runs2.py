"""build_all_runs.py"""

# First, traverse the `images_evaluation_supervised` and keep a record of the training and test characers, grouped
# by the class. Repeat this with the unseen classes by traversing the `images_evaluation_unseen` directory.

# Next, iterate through `num_runs` and for each run N, do the following steps:
# - Pick J (=19) seen classes, K (=1) unseen classes
# - Copy all training characters of J seen classes into `runN/supervised/train`
# - Copy all test characters of J seen classes into `runN/supervised/test`
# - Copy all characters of K unseen classes into `run/unseen`
# - Randomly sample characters from the above, based on (J:K) ratio for one-shot classification
# - Delete the used classes from the class map
# - REPEAT

import os
import random
import logging

from pathlib import Path
from shutil import copyfile


OUTPUT_PATH = './data/omniglot/all_runs_lakelike'
unseen_image_folder = './data/omniglot/images_evaluation_unseen'
supervised_image_folder = './data/omniglot/images_evaluation_supervised'

SEED = 50
IGNORE_LIST = ['.DS_Store']

UNSEEN_CLASS_MAP = {}
SUPERVISED_CLASS_MAP = {}

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)


for subset in os.listdir(supervised_image_folder):
  if subset in IGNORE_LIST:
    continue

  if os.path.isdir(os.path.join(supervised_image_folder, subset)):
    if subset not in SUPERVISED_CLASS_MAP:
      SUPERVISED_CLASS_MAP[subset] = {}

    for family in os.listdir(os.path.join(supervised_image_folder, subset)):
      if family in IGNORE_LIST:
        continue

      if os.path.isdir(os.path.join(supervised_image_folder, subset)) and family not in SUPERVISED_CLASS_MAP[subset]:

        for character in os.listdir(os.path.join(supervised_image_folder, subset, family)):
          character_folder = os.path.join(supervised_image_folder, subset, family, character)

          if os.path.isdir(character_folder):
            character_files = os.listdir(character_folder)
            character_label = int(character_files[0].split('_')[0])

            SUPERVISED_CLASS_MAP[subset][character_label] = []

            for character_file in character_files:
              character_filepath = os.path.join(character_folder, character_file)

              SUPERVISED_CLASS_MAP[subset][character_label].append(character_filepath)
      else:
        logging.warning('Path to character is not a directory: %s',
                        os.path.join(supervised_image_folder, subset, family))
  else:
    logging.warning('Path to subset is not a directory: %s', os.path.join(supervised_image_folder, subset))


for family in os.listdir(os.path.join(unseen_image_folder)):
  if family in IGNORE_LIST:
    continue

  if os.path.isdir(os.path.join(unseen_image_folder)) and family not in UNSEEN_CLASS_MAP:
    for character in os.listdir(os.path.join(unseen_image_folder, family)):
      character_folder = os.path.join(unseen_image_folder, family, character)

      if os.path.isdir(character_folder):
        character_files = os.listdir(character_folder)
        character_label = int(character_files[0].split('_')[0])

        UNSEEN_CLASS_MAP[character_label] = []

        for character_file in character_files:
          character_filepath = os.path.join(character_folder, character_file)

          UNSEEN_CLASS_MAP[character_label].append(character_filepath)
  else:
    logging.warning('Path to alphabet is not a directory: %s',
                    os.path.join(unseen_image_folder, family))

num_runs = 20
num_unseen = 1
num_seen = 19

ALL_RUNS = {}

for run_idx in range(1, num_runs + 1):
  run_folder = 'run' + str(run_idx).zfill(2)

  ALL_RUNS[run_folder] = {
      'supervised': {
          'training': [],
          'test': []
      },
      'unseen': [],
      'oneshot': {
          'training': [],
          'test': []
      }
  }

  seen_classes = random.sample(list(SUPERVISED_CLASS_MAP['train']), num_seen)
  unseen_classes = random.sample(list(UNSEEN_CLASS_MAP.keys()), num_unseen)

  for unseen_class in unseen_classes:
    unseen_chars = UNSEEN_CLASS_MAP[unseen_class]

    ALL_RUNS[run_folder]['unseen'].extend(unseen_chars)

    oneshot_unseen_samples = random.sample(unseen_chars, 2)
    oneshot_unseen_train_sample = [oneshot_unseen_samples[0]]
    oneshot_unseen_test_sample = [oneshot_unseen_samples[1]]

    ALL_RUNS[run_folder]['oneshot']['test'].extend(oneshot_unseen_train_sample)
    ALL_RUNS[run_folder]['oneshot']['training'].extend(oneshot_unseen_test_sample)

    del UNSEEN_CLASS_MAP[unseen_class]

  for seen_class in seen_classes:
    train_chars = SUPERVISED_CLASS_MAP['train'][seen_class]
    test_chars = SUPERVISED_CLASS_MAP['test'][seen_class]

    ALL_RUNS[run_folder]['supervised']['training'].extend(train_chars)
    ALL_RUNS[run_folder]['supervised']['test'].extend(test_chars)

    random_char = random.sample(range(len(train_chars)), 1)[0]

    oneshot_seen_train_sample = random.sample(train_chars, 1)
    oneshot_seen_test_sample = random.sample(test_chars, 1)

    ALL_RUNS[run_folder]['oneshot']['test'].extend(oneshot_seen_train_sample)
    ALL_RUNS[run_folder]['oneshot']['training'].extend(oneshot_seen_test_sample)

    del SUPERVISED_CLASS_MAP['train'][seen_class]
    del SUPERVISED_CLASS_MAP['test'][seen_class]


for run_folder in ALL_RUNS:
  run_folder_path = os.path.join(OUTPUT_PATH, run_folder)
  Path(run_folder_path).mkdir(parents=True, exist_ok=True)

  unseen_folder_path = os.path.join(run_folder_path, 'unseen')
  Path(unseen_folder_path).mkdir(parents=True, exist_ok=True)

  oneshot_train_folder_path = os.path.join(run_folder_path, 'oneshot', 'training')
  oneshot_test_folder_path = os.path.join(run_folder_path, 'oneshot', 'test')
  Path(oneshot_train_folder_path).mkdir(parents=True, exist_ok=True)
  Path(oneshot_test_folder_path).mkdir(parents=True, exist_ok=True)

  supervised_train_folder_path = os.path.join(run_folder_path, 'supervised', 'training')
  supervised_test_folder_path = os.path.join(run_folder_path, 'supervised', 'test')
  Path(supervised_train_folder_path).mkdir(parents=True, exist_ok=True)
  Path(supervised_test_folder_path).mkdir(parents=True, exist_ok=True)

  def copy_files(src, trg, rename=False):
    for i, char_path in enumerate(src):
      filename = os.path.basename(char_path)

      if rename:
        file_label = int(filename.split('_')[0])
        filename = str(i + 1).zfill(2) + '_' + str(file_label) + '.png'

      copyfile(char_path, os.path.join(trg, filename))

  copy_files(ALL_RUNS[run_folder]['unseen'], unseen_folder_path)

  copy_files(ALL_RUNS[run_folder]['oneshot']['training'], oneshot_train_folder_path, rename=True)
  copy_files(ALL_RUNS[run_folder]['oneshot']['test'], oneshot_test_folder_path, rename=True)

  copy_files(ALL_RUNS[run_folder]['supervised']['training'], supervised_train_folder_path)
  copy_files(ALL_RUNS[run_folder]['supervised']['test'], supervised_test_folder_path)
