# For each run:
# - Pick 1 file from each parent dir (alphabet) for train
# - Pick 1 file from each parent dir (alphabet) for test

import os
import random
import logging

from pathlib import Path
from shutil import copyfile


OUTPUT_PATH = './data/omniglot/all_runs_unseen'
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
    append_family = False
    if subset not in SUPERVISED_CLASS_MAP:
      SUPERVISED_CLASS_MAP[subset] = {}
      append_family = True

    for family in os.listdir(os.path.join(supervised_image_folder, subset)):
      if family in IGNORE_LIST:
        continue

      if os.path.isdir(os.path.join(supervised_image_folder, subset)):
        append_characters = False
        if family not in SUPERVISED_CLASS_MAP[subset]:
          SUPERVISED_CLASS_MAP[subset][family] = {}
          append_characters = True

          for character in os.listdir(os.path.join(supervised_image_folder, subset, family)):
            character_folder = os.path.join(supervised_image_folder, subset, family, character)
            if os.path.isdir(character_folder):
              character_files = os.listdir(character_folder)
              character_label = int(character_files[0].split('_')[0])
              SUPERVISED_CLASS_MAP[subset][family][character] = character_files
      else:
        logging.warning('Path to alphabet is not a directory: %s',
                        os.path.join(supervised_image_folder, subset, family))
  else:
    logging.warning('Path to subset is not a directory: %s', os.path.join(supervised_image_folder, subset))

for family in os.listdir(unseen_image_folder):
  if family in IGNORE_LIST:
    continue

  if os.path.isdir(os.path.join(unseen_image_folder, family)):
    append_characters = False
    if family not in UNSEEN_CLASS_MAP:
      UNSEEN_CLASS_MAP[family] = {}
      append_characters = True

      for character in os.listdir(os.path.join(unseen_image_folder, family)):
        character_folder = os.path.join(unseen_image_folder, family, character)
        if os.path.isdir(character_folder):
          character_files = os.listdir(character_folder)
          character_label = int(character_files[0].split('_')[0])
          UNSEEN_CLASS_MAP[family][character] = character_files
  else:
    logging.warning('Path to alphabet is not a directory: %s', os.path.join(unseen_image_folder, family))

num_runs = 20
num_unseen = 1
num_seen = 19

ALL_RUNS = {}

for run_idx in range(1, num_runs + 1):
  run_folder = 'run' + str(run_idx).zfill(2)

  ALL_RUNS[run_folder] = {
      'training': [],
      'test': []
  }

  seen_alphabets = random.sample(list(UNSEEN_CLASS_MAP), num_seen)
  unseen_alphabets = random.sample(list(UNSEEN_CLASS_MAP), num_unseen)

  for alphabet in unseen_alphabets:
    unseen_chars = UNSEEN_CLASS_MAP[alphabet]
    random_char = random.sample(list(unseen_chars), 1)[0]

    unseen_samples = unseen_chars[random_char]

    train_sample = unseen_samples.pop(random.randrange(len(unseen_samples)))
    test_sample = unseen_samples.pop(random.randrange(len(unseen_samples)))

    train_sample_path = os.path.join(unseen_image_folder, alphabet, random_char, train_sample)
    test_sample_path = os.path.join(unseen_image_folder, alphabet, random_char, test_sample)

    ALL_RUNS[run_folder]['training'].append(train_sample_path)
    ALL_RUNS[run_folder]['test'].append(test_sample_path)

  for alphabet in seen_alphabets:
    train_chars = SUPERVISED_CLASS_MAP['train'][alphabet]
    test_chars = SUPERVISED_CLASS_MAP['test'][alphabet]

    random_char = random.sample(list(train_chars), 1)[0]

    train_samples = train_chars[random_char]
    test_samples = test_chars[random_char]

    train_sample = train_samples.pop(random.randrange(len(train_samples)))
    test_sample = test_samples.pop(random.randrange(len(test_samples)))

    train_sample_path = os.path.join(supervised_image_folder, 'train', alphabet, random_char, train_sample)
    test_sample_path = os.path.join(supervised_image_folder, 'test', alphabet, random_char, test_sample)

    ALL_RUNS[run_folder]['training'].append(train_sample_path)
    ALL_RUNS[run_folder]['test'].append(test_sample_path)

for run_folder in ALL_RUNS:
  run_folder_path = os.path.join(OUTPUT_PATH, run_folder)
  Path(run_folder_path).mkdir(parents=True, exist_ok=True)

  train_folder_path = os.path.join(run_folder_path, 'training')
  Path(train_folder_path).mkdir(parents=True, exist_ok=True)

  for i, char_path in enumerate(ALL_RUNS[run_folder]['training']):
    filename = os.path.basename(char_path)
    character_label = int(filename.split('_')[0])
    new_filename = str(i + 1).zfill(2) + '_' + str(character_label) + '.png'
    copyfile(char_path, os.path.join(train_folder_path, new_filename))

  test_folder_path = os.path.join(run_folder_path, 'test')
  Path(test_folder_path).mkdir(parents=True, exist_ok=True)

  for i, char_path in enumerate(ALL_RUNS[run_folder]['test']):
    filename = os.path.basename(char_path)
    character_label = int(filename.split('_')[0])
    new_filename = str(i + 1).zfill(2) + '_' + str(character_label) + '.png'
    copyfile(char_path, os.path.join(test_folder_path, new_filename))
