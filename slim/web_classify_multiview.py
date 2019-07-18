# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""
import base64
import math
import os
import cv2
import tensorflow as tf
import sys
import threading
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops
import numpy as np


slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from datasets import species_big
import preprocessing.species_big_preprocessing as species_big_preprocessing

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
  return tf.select(-tf.random_uniform(grad.get_shape(), minval=0, maxval=0.05) < grad,
                   gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

def normalise(inp):
  return (inp - tf.reduce_min(inp)) / (tf.reduce_max(inp) - tf.reduce_min(inp))


def extract_saliency_img(saliency):
  saliency_identity = (tf.squeeze(saliency))
  saliency_identity = tf.maximum(saliency_identity, 0)
  saliency_identity = normalise(saliency_identity)  # * images[0])
  saliency_identity = tf.scalar_mul(255, saliency_identity)
  saliency_identity = tf.clip_by_value(saliency_identity, -255, 255)
  return saliency_identity


def extract_rgb_image(images):
  img_identity = tf.identity(tf.squeeze(images))
  img_identity = tf.add(1.0, img_identity)
  img_identity = tf.clip_by_value(img_identity, 0.0, 2.0)
  img_identity = tf.scalar_mul(127.4, img_identity)
  img_identity = tf.clip_by_value(img_identity, 0.0, 255)
  return img_identity

def feed_iterator():
    f = open(os.path.expanduser('~/classify_pipe.fifopipe'), 'r')
    lines = f.readlines()
    ls = []
    for line in lines:
      ls.extend([(base64.standard_b64decode(x[0]), int(x[1].split('|')[0]), x[1].split('|')[1]) for x in
                 [x.split('>>>INDEX<<<') for x in line.split(">>>EOF<<<")[:-1]]])

    for img, idx, priors in ls:
      try:
        img_np = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
        conv_img_bytes = bytes(cv2.imencode('.jpeg', img_np)[1])
        yield conv_img_bytes, idx, priors
      except Exception as e:
        print('bad image passed:', e, file=sys.stderr)
        continue

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Session() as sess:
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
      tf_global_step = slim.get_or_create_global_step()

      ######################
      # Select the dataset #
      ######################
      dataset = species_big.SpeciesDataset(FLAGS.current_level).get_split('validation', '')

      print(FLAGS.current_level)
      print(dataset.num_classes)

      ####################
      # Select the model #
      ####################
      func = nets_factory.inception.inception_resnet_v2_multiview

      def network_fn(main, side):
        arg_scope = nets_factory.inception.inception_resnet_v2_multiview_arg_scope()
        with slim.arg_scope(arg_scope):
          return func(main, side, num_classes=(dataset.num_classes), is_training=False)

      imgbytes, indexes, priors = tf.placeholder(tf.string), tf.placeholder(tf.int32), tf.placeholder(tf.string)

      image = tf.image.decode_jpeg(imgbytes, channels=3)

      eval_image_size = FLAGS.eval_image_size or 299

      image_main, image_side = species_big_preprocessing.preprocess_for_eval_multi(image, eval_image_size,
                                                                                   eval_image_size)
      images_main = tf.expand_dims(image_main, 0)
      images_side = tf.expand_dims(image_side, 0)

      logits, endpoints = network_fn(images_main, images_side)

      if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
      else:
        variables_to_restore = slim.get_variables_to_restore()

      saver = tf.train.Saver(variables_to_restore)


      #### Calculating guided backprop.
      squeezed_logit = tf.squeeze(logits)
      bottleneck = endpoints['Logits'][0]  # logits[0] #endpoints['mixed_8x8x2048b']
      in_var = endpoints['Inputs']  # tf.Variable(0.)
      vals, indices = tf.nn.top_k(bottleneck, k=10)  # tf.reduce_max(bottleneck)
      features = endpoints['Mixed_7a']
      maximisers = tf.gradients([squeezed_logit[indices[0]]], [features]) * features
      maximisers = tf.squeeze(maximisers)
      # print(maximisers)
      feat_sum = tf.reduce_sum(maximisers, [0, 1])
      # print('featsum', feat_sum)
      _, maxindices = tf.nn.top_k(feat_sum, k=10)

      func = tf.gather(tf.reduce_sum(features[0], [0, 1]),
                       maxindices)  # tf.reduce_sum(tf.slice(features[0], [0,0,maxindices[9]], [17,17,1])) #bottleneck[indices[0]]# - tf.reduce_mean(bottleneck)# + bottleneck[indices[1]] + bottleneck[indices[2]] # tf.reduce_max(bottleneck) # (tf.reduce_sum(bottleneck)) # tf.reshape(tf.one_hot(tf.argmax(bottleneck, 1), num_classes), [1, num_classes]) #bottleneck #tf.reduce_min(bottleneck)
      reduced_feats = tf.reduce_sum(features[0], [0, 1])
      saliency_single = tf.gradients([func], [in_var])

      def s_for_feat(feats, feat_idx):
        return normalise(tf.maximum(0.0, tf.gradients([feats[feat_idx]], [in_var])[0]))  # * feat_sum[feat_idx]

      saliency = s_for_feat(reduced_feats, maxindices[0]) + s_for_feat(reduced_feats, maxindices[1]) + s_for_feat(
        reduced_feats, maxindices[2]) + s_for_feat(reduced_feats, maxindices[3])

      saliency_identity = extract_saliency_img(saliency)
      image_identity = extract_rgb_image(images_side)

      out_op = tf.nn.softmax(logits)
      labels = indexes

      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
      else:
        checkpoint_path = FLAGS.checkpoint_path

      # if os.path.isabs(checkpoint_path):
        # Restores from checkpoint with absolute path.
      saver.restore(sess, checkpoint_path)
      # else:
      #   Restores from checkpoint with relative path.
        # saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
        #                                  ))

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        print('starting eval...')

        while not coord.should_stop():
            for feed_img_bytes, feed_index, feed_priors in feed_iterator():

              out, img_label, img, priors_out, saliency = sess.run(
                      [out_op, tf.identity(labels), image_identity, tf.identity(priors), saliency_identity], feed_dict={imgbytes: feed_img_bytes, indexes: feed_index, priors: feed_priors})
              reconstructed = np.array(img, dtype=np.uint8)

              saliency_img = np.array(saliency, dtype=np.uint8)

              saliency_img = cv2.cvtColor(saliency_img, cv2.COLOR_RGB2BGR)
              # grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

              reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)

              print("begin\n")
              print(">" + ",".join(['%.5f' % float(num) for num in out[0]]) + "|" + str(int(img_label)) + "|" + str(
                base64.standard_b64encode(bytes(cv2.imencode('.jpeg', reconstructed)[1])), 'utf-8') + "|" + str(priors_out,
                                                                                                                'utf-8') + "|" + str(
                base64.standard_b64encode(bytes(cv2.imencode('.jpeg', saliency_img)[1])), 'utf-8'))
              print("\nend")
              sys.stdout.flush()

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()

if __name__ == '__main__':
  tf.app.run()
