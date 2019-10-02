import tensorflow as tf
import numpy as np

from aha.components.diff_plasticity_component import DifferentiablePlasticityComponent


class DifferentiablePlasticityUnitTests(tf.test.TestCase):

  def setup_simple_session(self, dp,
                           batch_size=10, width=4, height=4,
                           sample_repeats=1, blank_repeats=0, pres_repeats=0, degrade=False, bias=False):

    with self.test_session() as sess:

      hparams = DifferentiablePlasticityComponent.default_hparams()
      hparams.batch_size = batch_size
      hparams.filters = width * height
      hparams.bias = bias
      hparams.bt_sample_repeat = sample_repeats
      hparams.bt_blank_repeat = blank_repeats
      hparams.bt_degrade = degrade
      hparams.bt_presentation_repeat = pres_repeats

      batch_shape = [batch_size, width * height]
      batch_pl = tf.placeholder(tf.float32, batch_shape)

      dp.build(batch_pl, batch_shape, hparams)
      sess.run(tf.global_variables_initializer())

      feed_dict = {
        batch_pl: np.random.random(batch_shape)
      }

      dp.update_feed_dict(feed_dict, 'training')

      return sess, feed_dict, batch_pl

  def testVariablesTrain(self):
    with self.test_session():

      dp = DifferentiablePlasticityComponent()
      sess, feed_dict, _ = self.setup_simple_session(dp, degrade=True, bias=False)

      variables = tf.trainable_variables()
      print("Trainable Variables:")
      for v in variables:
        print(v)

      before = sess.run(tf.trainable_variables())
      _ = sess.run(dp.get_dual().get_op('training'), feed_dict=feed_dict)
      after = sess.run(tf.trainable_variables())
      i = 0
      for b, a in zip(before, after):
        print("variable - " + str(i))
        assert (b != a).any()  # Make sure something changed.
        i = i + 1

  def testBatchTransformer(self):
    with self.test_session():

      sample_repeats = 3
      blank_repeats = 2
      pres_repeats = 2
      degraded = True

      dp = DifferentiablePlasticityComponent()
      sess, feed_dict, batch_pl = self.setup_simple_session(dp,
                                                            batch_size=4, width=2, height=2,
                                                            sample_repeats=sample_repeats, blank_repeats=blank_repeats,
                                                            pres_repeats=2, degrade=degraded)

      batch = feed_dict[batch_pl]

      fetches = [dp.get_dual().get_op('bt_input'), dp.get_dual().get_op('bt_output')]
      bt_input, bt_output = sess.run(fetches, feed_dict=feed_dict)

      # has used an uncorrupted version of input
      assert np.allclose(bt_input, batch)

      # output is the correct size
      batch_length = batch.shape[0]
      output_length = pres_repeats * batch_length * (sample_repeats + blank_repeats) + (2 if degraded else 0)

      print("Batch:\n", batch)
      print("Output:\n", bt_output)

      assert bt_output.shape[0] == output_length      # inserted the correct number of additional elements
      assert bt_output.shape[1:] == batch.shape[1:]   # the shape of the samples is correct

      blank_sample = np.zeros(batch.shape[1:])

      pres_length = batch_length * (sample_repeats + blank_repeats)
      for p_idx in range(pres_repeats):
        for x_idx in range(batch_length):
          for r in range(sample_repeats):
            y_idx = (p_idx * pres_length) + x_idx * (sample_repeats + blank_repeats) + r
            assert np.allclose(batch[x_idx], bt_output[y_idx])     # sample has been repeated

          for b in range(blank_repeats):
            y_idx = x_idx * (sample_repeats + blank_repeats) + sample_repeats + b
            assert np.allclose(bt_output[y_idx], blank_sample)

      degraded = bt_output[-2]
      target = bt_output[-1]
      print("Degraded\n", degraded)
      print("Target:\n", target)

      # TODO check the number of zeros is half the length in degraded


if __name__ == '__main__':
  tf.test.main()
