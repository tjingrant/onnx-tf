from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import run_model, prepare
from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestNode(unittest.TestCase):
  MODEL_PATH = "../../onnx_models/"

  def test_squeezenet(self):
    model = onnx.load(self.MODEL_PATH + "squeezenet/model.pb")
    tf_rep = prepare(model)
    for i in range(3):
      sample = np.load(self.MODEL_PATH + "squeezenet/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_almost_equal(outputs[0], my_out['softmaxout_1'], decimal=4)

  def test_vgg16(self):
    model = onnx.load(self.MODEL_PATH + "vgg16/model.pb")
    tf_rep = prepare(model)
    for i in range(3):
      sample = np.load(self.MODEL_PATH + "vgg16/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_almost_equal(outputs[0], my_out['gpu_0/softmax_1'], decimal=4)

  def test_vgg19(self):
    model = onnx.load(self.MODEL_PATH + "vgg19/model.pb")
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "vgg19/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_almost_equal(outputs[0], my_out['prob_1'], decimal=4)

if __name__ == '__main__':
  unittest.main()
