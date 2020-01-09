"""Parse AHA results from Jenkins output."""

input_path = 'jenkins.log'
output_path = 'summary.csv'


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

for i, j in zip(headings, values):
  print(data[i])
  print(data[j])
  print('\n')

with open(output_path, 'w') as f:
  f.write(data[headings[0]])
  f.write('\n')

  for value in values:
    f.write(data[value])
    f.write('\n')
