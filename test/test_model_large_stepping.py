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

def find_between(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

class TestNode(unittest.TestCase):
  MODEL_PATH = "../../onnx_models/"

  # def test_squeezenet(self):
  #   model = onnx.load(self.MODEL_PATH + "squeezenet/model.pb")
  #   tf_rep = prepare(model)
  #   for i in range(3):
  #     sample = np.load(self.MODEL_PATH + "squeezenet/test_data_{}.npz".format(str(i)), encoding='bytes')

  #     inputs = list(sample['inputs'])
  #     outputs = list(sample['outputs'])

  #     my_out = tf_rep.run(inputs)
  #     np.testing.assert_almost_equal(outputs[0], my_out['softmaxout_1'], decimal=4)

  # def test_vgg16(self):
  #   model = onnx.load(self.MODEL_PATH + "vgg16/model.pb")
  #   tf_rep = prepare(model)
  #   for i in range(3):
  #     sample = np.load(self.MODEL_PATH + "vgg16/test_data_{}.npz".format(str(i)), encoding='bytes')

  #     inputs = list(sample['inputs'])
  #     outputs = list(sample['outputs'])

  #     my_out = tf_rep.run(inputs)
  #     np.testing.assert_almost_equal(outputs[0], my_out['gpu_0/softmax_1'], decimal=4)

  # def test_vgg19(self):
  #   model = onnx.load(self.MODEL_PATH + "vgg19/model.pb")
  #   tf_rep = prepare(model)

  #   for i in range(3):
  #     sample = np.load(self.MODEL_PATH + "vgg19/test_data_{}.npz".format(str(i)), encoding='bytes')

  #     inputs = list(sample['inputs'])
  #     outputs = list(sample['outputs'])

  #     my_out = tf_rep.run(inputs)
  #     np.testing.assert_almost_equal(outputs[0], my_out['prob_1'], decimal=4)


  # def test_bvlc_alexnet(self):
  #   model = onnx.load(self.MODEL_PATH + "bvlc_alexnet/model.pb")
  #   tf_rep = prepare(model)

  #   for i in range(3):
  #     sample = np.load(self.MODEL_PATH + "bvlc_alexnet/test_data_{}.npz".format(str(i)), encoding='bytes')

  #     inputs = list(sample['inputs'])
  #     outputs = list(sample['outputs'])

  #     my_out = tf_rep.run(inputs)
  #     np.testing.assert_allclose(outputs[0], my_out['prob_1'], rtol=1e-1)

  def test_densenet(self):
    _model = onnx.load(self.MODEL_PATH + "inception_v1/model.pb")
    node_count = len(_model.graph.node)
    more_outputs = []
    output_to_check = []
    for node in _model.graph.node:
      more_outputs.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, (100, 100)))
      output_to_check.append(node.output[0])
    _model.graph.output.extend(more_outputs)

    tf_rep = prepare(_model)
    cf_rep = prepare2(_model)

    sample = np.load(self.MODEL_PATH + "inception_v1/test_data_{}.npz".format(str(1)), encoding='bytes')
    inputs = list(sample['inputs'])
    outputs = list(sample['outputs'])

    my_out = tf_rep.run(inputs)
    cf_out = cf_rep.run(inputs)

    for op in output_to_check:
      print("+"*50)
      print(op)
      try:
        np.savetxt(op.replace("/", "__") + ".cf", cf_out[op].flatten(), delimiter='\t')
        np.savetxt(op.replace("/", "__") + ".tf", my_out[op].flatten(), delimiter='\t')
        np.testing.assert_allclose(my_out[op], cf_out[op], rtol=1e-2)
        print("results of this layer are correct within tolerence.")
      except Exception as e:
        np.set_printoptions(threshold=np.inf)
        mismatch_percent = (find_between(str(e), "(mismatch", "%)"))
        print("mismatch with percentage {} %".format(mismatch_percent))
      print("="*50)

    # # print(model.graph.output)
    # # print(model.graph.node)
    # # model.graph.node = model.graph.node[:-1]
    # for i in range(1, node_count):
    #   model = onnx.load(self.MODEL_PATH + "bvlc_alexnet/model.pb")
    #   del model.graph.output[:]
    #   del model.graph.node[i+1:]
    #   output = helper.make_tensor_value_info(model.graph.node[i].output[0], TensorProto.FLOAT, (100, 100))
    #   print("current name " + output.name, " ", str(i))
    #   model.graph.output.extend([output])
    #   tf_rep = prepare(model)
    #   cf_rep = prepare2(model)
    #   for i in range(1):
    #     sample = np.load(self.MODEL_PATH + "bvlc_alexnet/test_data_{}.npz".format(str(i)), encoding='bytes')

    #     inputs = list(sample['inputs'])
    #     outputs = list(sample['outputs'])

    #     my_out = tf_rep.run(inputs)
    #     with open(output.name + ".tf",'w') as f:
    #       np.savetxt(f, my_out[output.name].flatten(), delimiter='\t')
    #     cf_out = cf_rep.run(inputs)
    #     with open(output.name + ".cf",'w') as f:
    #       np.savetxt(f, cf_out[output.name].flatten(), delimiter='\t')

    #     try:
    #       np.testing.assert_allclose(my_out[output.name], cf_out[output.name], rtol=1e-2)
    #     except Exception as e:
    #       np.set_printoptions(threshold=np.inf)
    #       print(e)
    #   del model
    #   del tf_rep
    #   del cf_rep

if __name__ == '__main__':
  unittest.main()
  pass

# # (1, 1024, 6, 6)
# cf = np.loadtxt("inception_5b__output_1.cf", delimiter='\t').reshape([1, 1024, 6, 6])
# cf_res = np.loadtxt("pool5__7x7_s1_1.cf", delimiter='\t').reshape([1024])
# np.testing.assert_allclose(np.average(cf, axis=(2,3)).reshape([1024]), cf_res)

# tf = np.loadtxt("inception_5b__output_1.tf", delimiter='\t').reshape([1, 1024, 6, 6])
# tf_res = np.loadtxt("pool5__7x7_s1_1.tf", delimiter='\t').reshape([1024])
# np.testing.assert_allclose(np.average(tf, axis=(2,3)).reshape([1024]), tf_res)



