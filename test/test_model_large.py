from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import caffe2.python

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from onnx_caffe2.backend import prepare as prepare2
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
      np.testing.assert_allclose(outputs[0], my_out['softmaxout_1'], rtol=1e-3)

  def test_vgg16(self):
    model = onnx.load(self.MODEL_PATH + "vgg16/model.pb")
    tf_rep = prepare(model)
    for i in range(3):
      sample = np.load(self.MODEL_PATH + "vgg16/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_allclose(outputs[0], my_out['gpu_0/softmax_1'], rtol=1e-3)

  def test_vgg19(self):
    model = onnx.load(self.MODEL_PATH + "vgg19/model.pb")
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "vgg19/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_allclose(outputs[0], my_out['prob_1'], rtol=1e-3)

  def test_bvlc_alexnet(self):
    model = onnx.load(self.MODEL_PATH + "bvlc_alexnet/model.pb")
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "bvlc_alexnet/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.testing.assert_allclose(outputs[0], my_out['prob_1'], rtol=1e-3)

  def test_shuffle_net(self):
    return
    model = onnx.load(self.MODEL_PATH + "shufflenet/model.pb")
    print(model.graph.output)
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "shufflenet/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.savetxt('test.out', my_out['gpu_0/softmax_1'], delimiter='\t')
      np.savetxt('ref.out', outputs[0], delimiter='\t')
      np.testing.assert_almost_equal(outputs[0], my_out['gpu_0/softmax_1'], decimal=5)

  def test_dense_net(self):
    model = onnx.load(self.MODEL_PATH + "densenet121/model.pb")
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "densenet121/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.savetxt('test.out', my_out['fc6_1'], delimiter='\t')
      np.savetxt('ref.out', outputs[0], delimiter='\t')
      np.testing.assert_allclose(outputs[0], my_out['fc6_1'], rtol=1e-3)

  def test_resnet50(self):
    model = onnx.load(self.MODEL_PATH + "resnet50/model.pb")
    tf_rep = prepare(model)

    for i in range(3):
      sample = np.load(self.MODEL_PATH + "resnet50/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      my_out = tf_rep.run(inputs)
      np.savetxt('test.out', my_out['gpu_0/softmax_1'], delimiter='\t')
      np.savetxt('ref.out', outputs[0], delimiter='\t')
      np.testing.assert_allclose(outputs[0], my_out['gpu_0/softmax_1'], rtol=1e-3)


if __name__ == '__main__':
  unittest.main()
