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

import numpy as np
import tensorflow as tf
import random
import logging
import sys
import time
import os
from os.path import dirname, abspath

d = dirname(dirname(abspath(__file__)))  # Each dirname goes up one
sys.path.append(d)
sys.path.append(os.path.join(d, 'utils'))
import generic_utils

sys.path.append(os.path.join(d, 'components'))

import diff_plasticity_int_tests

PATTERNSIZE = 100        # : sample_size
NBNEUR = PATTERNSIZE + 0  # NbNeur = Pattern Size + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
# ETA = .01               # The "learning rate" of plastic connections - we actually learn it
ADAMLEARNINGRATE = 3e-4   # The learning rate of the Adam optimizer
RNGSEED = 1  # Initial random seed - can be modified by passing a number as command-line argument

# Note that these patterns are likely not optimal
PROBADEGRADE = .5   # Proportion of bits to zero out in the target pattern at test time  : degrade_factor
NBPATTERNS = 5      # The number of patterns to learn in each episode     : number_samples
NBPRESCYCLES = 2    # Number of times each pattern is to be presented   : presentation_repeats
PRESTIME = 6        # Number of time steps for each presentation            : sample_repeats
PRESTIMETEST = 6    # Same thing but for the final test pattern         : degraded_repeats
INTERPRESDELAY = 4  # Duration of zero-input interval between presentations   : blank_repeats
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode


def reset_random(seed):
  np.random.seed(seed)
  random.seed(seed)
  tf.set_random_seed(seed)


if len(sys.argv) == 2:
  RNGSEED = int(sys.argv[1])
  print("Setting RNGSEED to " + str(RNGSEED))

np.set_printoptions(precision=3)
reset_random(RNGSEED)

ITNS = 1000
BPIT = True
LOAD_PARAMS_FROM_DISK = False
SPARSITY = 0.5  # fractional sparsity e.g. 0.5 = 0.5 active,   0.2 = 0.8 active

print_every = 10

DEBUG_GRAPH_PRINT = False
DEBUG_GRAPH_SUMM = True


def zero_if_less_than(x, eps):
  """Return 0 if x<eps, otherwise return x"""
  if x < eps:
    return 0
  else:
    return x


# Generate the full list of inputs for an episode.
# The inputs are returned as numpy array of shape = [NbSteps x NbNeur]
def generate_inputs_and_target():
  inputT = np.zeros((NBSTEPS, NBNEUR))

  # Create the random patterns to be memorized in an episode
  length_sparse = int(PATTERNSIZE * SPARSITY)
  seedp = np.ones(PATTERNSIZE)
  seedp[:length_sparse] = -1
  patterns = []
  for nump in range(NBPATTERNS):
    p = np.random.permutation(seedp)
    patterns.append(p)

  # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
  # Choosing the test pattern, partially zero'ed out, that the network will have to complete
  testpattern = random.choice(patterns).copy()
  preservedbits = np.ones(PATTERNSIZE)
  preservedbits[:int(PROBADEGRADE * PATTERNSIZE)] = 0
  np.random.shuffle(preservedbits)
  degradedtestpattern = testpattern * preservedbits

  logging.debug("test pattern     = ", testpattern)
  logging.debug("degraded pattern = ", degradedtestpattern)

  # Inserting the inputs in the input tensor at the proper places
  for nc in range(NBPRESCYCLES):
    np.random.shuffle(patterns)
    for ii in range(NBPATTERNS):
      for nn in range(PRESTIME):
        numi = nc * (NBPATTERNS * (PRESTIME + INTERPRESDELAY)) + ii * (PRESTIME + INTERPRESDELAY) + nn
        inputT[numi][:PATTERNSIZE] = patterns[ii][:]

  # Inserting the degraded pattern
  for nn in range(PRESTIMETEST):
    logging.debug("insert degraded pattern at: [{0},{1},:{2}]".format(-PRESTIMETEST + nn, 0, PATTERNSIZE))
    inputT[-PRESTIMETEST + nn][:PATTERNSIZE] = degradedtestpattern[:]

  for nn in range(NBSTEPS):
    inputT[nn][-1] = 1.0  # Bias neuron
    inputT[nn] *= 20.0  # Strengthen inputs

  logging.debug("shape of inputT: ", np.shape(inputT))

  return inputT, testpattern


# -----------------------------
#  build the graph
# -----------------------------


def tfsumsc(name="name", tensor=None):
  if DEBUG_GRAPH_SUMM:
    tf.summary.scalar(name=name, tensor=tensor)
  else:
    pass


def tfprint(var, message="", summarize=10, override=False):
  if override or DEBUG_GRAPH_PRINT:
    message = "\n" + message + "\n\t"
    return tf.Print(var, [var], message=message, summarize=summarize)
  else:
    return var


with tf.variable_scope("pl-inputs"):
  episodeX_pl = tf.placeholder(tf.float64, shape=[NBSTEPS, NBNEUR], name="epX")
  episodeT_pl = tf.placeholder(tf.float64, shape=[PATTERNSIZE], name="epT")
  init_state_pl = tf.placeholder(tf.float64, shape=[NBNEUR], name="init")
  init_hebb_pl = tf.placeholder(tf.float64, shape=[NBNEUR, NBNEUR], name="hebb_init")

  x_series = tf.unstack(episodeX_pl, axis=0, name="epX-x_series")  # NBSTEPS of [NBNEUR]

  current_state = tf.reshape(init_state_pl, shape=[1, NBNEUR], name="init-curr-state")
  hebb = init_hebb_pl

with tf.variable_scope("slow-weights"):
  w_default = 0.01
  alpha_default = 0.01
  eta_default = 0.01

  w = tf.get_variable(name="w", initializer=(w_default * np.random.rand(NBNEUR, NBNEUR)))
  alpha = tf.get_variable(name="alpha", initializer=(alpha_default * np.random.rand(NBNEUR, NBNEUR)))
  eta = tf.get_variable(name="eta", initializer=(eta_default * np.ones(shape=[1])))

with tf.variable_scope("layers"):
  hebb = tfprint(hebb, "*** initial hebb ***")
  current_state = tfprint(current_state, "*** initial state ***")
  w = tfprint(w, "*** w ***", override=False)
  alpha = tfprint(alpha, "*** alpha ***")

  next_state_ref = None
  x_input_ref = None
  i = 0
  for x_input in x_series:

    layer_name = "layer-" + str(i)
    with tf.variable_scope(layer_name):

      x_input = tfprint(x_input, str(i) + ": x_input")
      current_state = tfprint(current_state, str(i) + ": y(t-1)")

      # ---------- Calculate next output of the RNN
      if not DEBUG_GRAPH_PRINT:
        next_state = tf.tanh(tf.add(tf.matmul(current_state,
                                              tf.add(w, tf.multiply(alpha, hebb, name='lyr-mul'), name="lyr-add_w_ah"),
                                              name='lyr-mul-add-matmul')
                                    , x_input, "lyr_add_yxah_plus_x"), name="lyr-tanh")
      else:
        alpha_hebb = tf.multiply(alpha, hebb)
        alpha_hebb = tfprint(alpha_hebb, str(i) + ": *** alpha_hebb = multiply(alpha, hebb) ***")
        recurrent_input = tf.matmul(current_state, w + alpha_hebb)
        recurrent_input = tfprint(recurrent_input, str(i) + ": *** recc_in = y*(w+alpha_hebb) ***")
        tanharg = recurrent_input + x_input
        tanharg = tfprint(tanharg, str(i) + ": *** tanharg = recc_in + x_input ***")
        next_state = tf.tanh(tanharg, name="tanh")
        next_state = tfprint(next_state, str(i) + ": *** y(t) = tanh(recc_in + y) ***")

      with tf.variable_scope("fast_weights"):
        # ---------- Update Hebbian fast weights
        # outer product of (yin * yout) = (current_state * next_state)
        outer = tf.matmul(tf.reshape(current_state, shape=[NBNEUR, 1]), tf.reshape(next_state, shape=[1, NBNEUR]),
                          name="outer-product")
        outer = tfprint(outer, str(i) + ": *** outer = y(t-1) * y(t) ***")
        hebb = (1.0 - eta) * hebb + eta * outer
        hebb = tfprint(hebb, str(i) + ": *** hebb ***")  # , override=True)

      current_state = next_state
      i = i + 1

      next_state_ref = next_state
      x_input_ref = x_input

with tf.variable_scope("loss_calc"):
  # Compute loss for this episode (last step only)
  yy = tf.squeeze(current_state)
  # y = tf.slice(yy, begin=1, size=PATTERNSIZE)  # remove bias neuron for 'output' state
  y = yy[:PATTERNSIZE]  # remove bias neuron for 'output' state
  t = episodeT_pl

  loss = tf.reduce_sum(tf.square(tf.subtract(y, t, name="loss-diff"), name="loss-diff-sq"),
                       name="loss-diff-sq-sum")

with tf.variable_scope("summaries"):
  # tfsumsc(name="w-mean", tensor=tf.reduce_mean(w))
  # tfsumsc(name="expisodeX-sum", tensor=tf.reduce_sum( episodeX_pl))  # ~65,000 : (NBSTEPS-40) * (0.5*(NBNEUR-1)*20 + 20)
  #                                                                    # = 66*(10*101 + 20) = 66*1030 = ~67,980
  tfsumsc(name="x_input-sum",
          tensor=tf.reduce_sum(x_input_ref))  # should be the last input in the episode (degraded version)
  # tfsumsc(name="next_state-sum", tensor=tf.reduce_sum(next_state_ref))  #
  # tfsumsc(name="current_state-sum", tensor=tf.reduce_sum(current_state))
  tfsumsc(name="y-sum", tensor=tf.reduce_sum(y))  # y = current_state = next_state
  # tfsumsc(name="target-sum", tensor=tf.reduce_sum(episodeT_pl))  # 50
  tfsumsc(name="hebb-mean", tensor=tf.reduce_mean(hebb))

  tf.summary.scalar('loss-scalar', loss)

with tf.variable_scope("optimizer"):
  train_step = tf.train.AdamOptimizer(ADAMLEARNINGRATE).minimize(loss)

# -----------------------------
#  run the graph
# -----------------------------


with tf.Session() as sess:
  print("----------------------------------")
  print("Size of episode = " + str(NBSTEPS))
  print("Number of episodes = " + str(ITNS))
  print("Size of vector = " + str(PATTERNSIZE))
  print("Random seed = " + str(RNGSEED))
  print("----------------------------------")

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(generic_utils.get_summary_dir(), sess.graph)

  tf.global_variables_initializer().run()

  loss_list = []
  max_loss = 0
  total_loss = 0.0
  all_losses = []
  now_time = time.time()

  # present ITNS episodes
  init_state = np.zeros(NBNEUR)
  init_hebb = np.zeros(shape=[NBNEUR, NBNEUR])
  for numiter in range(ITNS):
    episodeX, target = diff_plasticity_int_tests.DiffPlasticityIntTests.generate_random_input(
      sample_size=PATTERNSIZE,
      number_samples=NBPATTERNS,
      presentation_repeats=NBPRESCYCLES,
      sample_repeats=PRESTIME,
      degraded_repeats=PRESTIMETEST,
      blank_repeats=INTERPRESDELAY,
      sparsity=SPARSITY,
      degrade_factor=PROBADEGRADE,
      add_bias=False
    )

    # reset_random(RNGSEED)
    # episodeX_org, target_org = generate_inputs_and_target()  # episodeX = [NBSTEPS x 1 x NBNEUR]
    #
    # assert (episodeX_org.size == episodeX.size), "Ep Original: \n" + str(episodeX_org) + "\nEp New: \n" + str(episodeX)
    # assert (episodeX_org == episodeX).all(), "Ep Original: \n" + str(episodeX_org) + "\nEp New: \n" + str(episodeX)
    # assert (target_org == target).all(), "Target: \n" + str(episodeX_org) + "\nTgt New: \n" + str(episodeX)

    _summary, _loss, _train_step, _y, _t, _w, _x = sess.run([merged, loss,
                                                             train_step,
                                                             y, t,
                                                             w, episodeX_pl],
                                                            feed_dict={
                                                              episodeX_pl: episodeX,  # [NBSTEPS, NBNEUR]
                                                              episodeT_pl: target,
                                                              init_state_pl: init_state,
                                                              init_hebb_pl: init_hebb})
    writer.add_summary(_summary, numiter)

    # Print statistics
    to = _t
    yo = _y[:PATTERNSIZE]  # in this version _y should be same size as yo already
    z = (np.sign(yo) != np.sign(to))
    loss_num = np.mean(z)  # Saved loss is the error rate
    total_loss += loss_num

    episodeX_sum = np.sum(episodeX)  # tb: input_x_sum    ~ 65,000
    target_sum = np.sum(_t)  # tb: target_sum     = 50
    y_sum = np.sum(_y)
    w_sum = np.sum(_w)

    # print("--- direct ---")
    # print("input_x_sum = " + str(episodeX_sum))
    # print("target_sum = " + str(target_sum))
    # print("loss = " + str(loss_num))
    # print("")
    #
    # print("--- from graph ---")
    # print("y_sum = " + str(y_sum))
    # print("w_sum = " + str(w_sum))
    # print("loss = " + str(_loss))
    # print("")

    # =========================================================
    # SANITY TESTS
    sanity_verbose = False
    if sanity_verbose:
      target_bt = _x
      print("Episode:\n")
      for i in range(_x.shape[0]):
        print(_x[i])
    # =========================================================

    if (numiter + 1) % print_every == 0:
      print((numiter, "===="))
      print("T", _t[:10])  # Target pattern to be reconstructed
      print("D", episodeX[-1][:10])  # Last input, degraded pattern fed to network at test time last num is bias neuron)
      print("Y", yo[:10])  # Final output of the network

      diff = yo - target[:]
      vfunc = np.vectorize(zero_if_less_than)
      vfunc(diff, 0.01)
      print("E", diff[:10])

      previous_time = now_time
      now_time = time.time()
      print("Time spent on last", print_every, "iters: ", now_time - previous_time)
      total_loss /= print_every
      all_losses.append(total_loss)
      print("Mean loss over last", print_every, "iters:", total_loss)
      print("loss op = " + str(_loss))
      print("")

      total_loss = 0
