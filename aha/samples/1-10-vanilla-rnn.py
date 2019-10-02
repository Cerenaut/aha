######################################################################################################
#
# Original from tutorial https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
#
# Changes:
#   - Updated to work with current Tensorflow
#   - Added name scopes for cleaner graph visualisation (not perfect, but good for our purposes)
#
# Intended Changes:
#   - Implement the same thing with tf dynamic rnn, then a duplicated version of their source
#   - Implement the FastWeights with the duplicated RNN modified to add the fast weights
#
######################################################################################################

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


def generate_data():
  x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0

  x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
  y = y.reshape((batch_size, -1))

  return x, y


def get_summary_dir(parent_dir='./run'):
  now = datetime.datetime.now()
  summary_dir = parent_dir + '/summaries_' + now.strftime("%Y%m%d-%H%M%S") + '/'
  return summary_dir

def tf_reduce_var(x, axis=None, keepdims=False):
  """Variance of a tensor, alongside the specified axis.
  Stolen from: https://stackoverflow.com/a/43409235
  # Arguments
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.
  # Returns
      A tensor with the variance of elements of `x`.
  """
  m = tf.reduce_mean(x, axis=axis, keepdims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def tf_build_stats_summaries(tensor, name_scope):
  """Build statistical summaries for a specific variable/tensor."""
  with tf.name_scope(name_scope):
    m_mean = tf.reduce_mean(tensor)
    m_var = tf_reduce_var(tensor)
    m_min = tf.reduce_min(tensor)
    m_max = tf.reduce_max(tensor)
    m_sum = tf.reduce_sum(tensor)

    mean_op = tf.summary.scalar('mean', m_mean)
    sd_op = tf.summary.scalar('sd', tf.sqrt(m_var))
    min_op = tf.summary.scalar('min', m_min)
    max_op = tf.summary.scalar('max', m_max)
    sum_op = tf.summary.scalar('sum', m_sum)

    stats_summaries = []
    stats_summaries.append(mean_op)
    stats_summaries.append(sd_op)
    stats_summaries.append(min_op)
    stats_summaries.append(max_op)
    stats_summaries.append(sum_op)

    return stats_summaries

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


with tf.name_scope("data"):
  batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
  batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

  init_state = tf.placeholder(tf.float32, [batch_size, state_size])

  # Unpack columns
  with tf.name_scope("input"):
    inputs_series = tf.unstack(batchX_placeholder, axis=1)
  with tf.name_scope("labels"):
    labels_series = tf.unstack(batchY_placeholder, axis=1)


with tf.name_scope("params"):
  with tf.name_scope("params1"):
    W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
    b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32
                    )

  with tf.name_scope("params2-logits"):
    W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)


with tf.name_scope("rnn"):

  # Forward pass
  current_state = init_state
  states_series = []
  i = 0
  for current_input in inputs_series:
    with tf.name_scope("reshape-" + str(i)):
      current_input = tf.reshape(current_input, [batch_size, 1])
    with tf.name_scope("concat-" + str(i)):
      input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns

    with tf.name_scope("layer-" + str(i)):
      next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
      states_series.append(next_state)
      current_state = next_state

    i = i + 1

with tf.name_scope("train"):
  with tf.name_scope("logit"):
    logits_series = [tf.matmul(state, W2) + b2 for state in states_series]  # Broadcasted addition

  with tf.name_scope("softmax"):
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

  with tf.name_scope("losses"):
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
              zip(logits_series, labels_series)]
    loss_tensor = tf.convert_to_tensor(losses)
    total_loss = tf.reduce_mean(losses)

    tf.summary.scalar('total_loss', total_loss)
    tf_build_stats_summaries(loss_tensor, 'loss_tensor')

    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
  plt.subplot(2, 3, 1)
  plt.cla()
  plt.plot(loss_list)

  for batch_series_idx in range(5):
    one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
    single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

    plt.subplot(2, 3, batch_series_idx + 2)
    plt.cla()
    plt.axis([0, truncated_backprop_length, 0, 2])
    left_offset = range(truncated_backprop_length)
    plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
    plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
    plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

  plt.draw()
  plt.pause(0.0001)


with tf.Session() as sess:

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(get_summary_dir(), sess.graph)

  tf.global_variables_initializer().run()

  plt.ion()
  plt.figure()
  plt.show()
  loss_list = []
  max_loss = 0

  for epoch_idx in range(num_epochs):
    x, y = generate_data()

    _current_state = np.zeros((batch_size, state_size))

    print("New data, epoch", epoch_idx)

    for batch_idx in range(num_batches):
      start_idx = batch_idx * truncated_backprop_length
      end_idx = start_idx + truncated_backprop_length

      batchX = x[:, start_idx:end_idx]
      batchY = y[:, start_idx:end_idx]

      summary, _total_loss, _train_step, _current_state, _predictions_series = sess.run(
        [merged, total_loss, train_step, current_state, predictions_series],
        feed_dict={
          batchX_placeholder: batchX,
          batchY_placeholder: batchY,
          init_state: _current_state
        })

      loss_list.append(_total_loss)
      if _total_loss > max_loss:
        max_loss = _total_loss

      idx = epoch_idx * num_batches + batch_idx
      writer.add_summary(summary, idx)


      if batch_idx % 100 == 0:
        print("Step", batch_idx, "Loss", _total_loss)
        print("\tmax loss = " + str(max_loss))
        plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
