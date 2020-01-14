"""Parse AHA results from Jenkins output."""

import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_path', type=str, help='an integer for the accumulator')

args = parser.parse_args()

if args.input_path is None or args.input_path == '' or not os.path.exists(args.input_path):
  raise ValueError('Input path does not exist.')

input_path = args.input_path
output_path = input_path[:-4] + '.csv'


with open(input_path, 'r') as f:
  data = f.readlines()

key = 'acc_mse_vc,'   # keyword to find the heading line
line_offset = 2       # how many lines apart are the headings and values

c = 0
values = []
headings = []

for num, line in enumerate(data):
  if key in line:
    headings.append(num)

  if c < len(headings) and num == headings[c] + line_offset:
    values.append(num)
    c += 1

# for i, j in zip(headings, values):
#   print(data[i])
#   print(data[j])
#   print('\n')

radius_increment = 0.05

with open(output_path, 'w') as f:
  f.write('radius,diameter,' + data[headings[0]])

  radius = 0
  for value in values:
    diameter = radius * 2
    f.write(str(radius) + ',' + str(diameter) + ',' + data[value])
    radius += radius_increment
