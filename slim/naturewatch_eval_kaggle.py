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

import pickle
import threading
import cv2
import os

import math
import tensorflow as tf

from nets import nets_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

from preprocessing import naturewatch_preprocessing

def setup_queue(sess):
  queue = tf.FIFOQueue(
    capacity=1000,
    dtypes=[tf.float32, tf.float32, tf.string])
  main_thread_graph = tf.get_default_graph()

  def runFeeder():
    print('RUNNIGN FEEDER')
    ls = os.listdir(FLAGS.dataset_dir)
    with main_thread_graph.as_default():
        for fname in ls:
          img_np = None

          try:
            img_np = cv2.imread(os.path.join(FLAGS.dataset_dir, fname), 1)
          except:
            sys.stderr.write('bad image passed\n')
            continue

          if img_np == None:
            sys.stderr.write('bad image passed\n')
            continue

          conv_img_bytes = bytes(cv2.imencode('.jpeg', img_np)[1])
          image = tf.image.decode_jpeg(tf.convert_to_tensor(conv_img_bytes, tf.string), channels=3)
          eval_image_size = 299
          image_main, image_side = naturewatch_preprocessing.preprocess_for_eval_multi(image, eval_image_size, eval_image_size)
          print(image_main.get_shape(), image_side.get_shape())
          # enqueues a (string, string)
          conv =  (image_main, image_side, tf.convert_to_tensor(fname, tf.string))

          enqueue_op = queue.enqueue(vals=conv)
          print(enqueue_op.graph)
          sess.run(enqueue_op)

  threading.Thread(target=runFeeder).start()
  return queue


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Session() as sess:
  # with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################

    ####################
    # Select the model #
    ####################
    func = nets_factory.inception.inception_resnet_v2_multiview

    def network_fn(main, side):
      arg_scope = nets_factory.inception.inception_resnet_v2_multiview_arg_scope()
      with slim.arg_scope(arg_scope):
        return func(main, side, num_classes=(5000), is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    queue = setup_queue(sess)
    print('queue is set up', queue)

    images_main, images_side, filenames = queue.dequeue_many(FLAGS.batch_size or 16)
    images_main.set_shape((16, 299, 299, 3)), images_side.set_shape((16, 299, 299, 3))
    # image = tf.image.decode_jpeg(imgbyte, channels=3)

    eval_image_size = FLAGS.eval_image_size or 299
    print(images_main)
    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images_main, images_side)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()
    
    saver = tf.train.Saver(variables_to_restore)
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

        # if os.path.isabs(checkpoint_path):
        # Restores from checkpoint with absolute path.
    saver.restore(sess, checkpoint_path)

    predictions = tf.argmax(logits, 1)
   
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        print('starting eval...')

        while not coord.should_stop():
          import time
          time.sleep(100)
          top_k, filename = sess.run([tf.nn.top_k(logits, 5), filenames])
          print(top_k)

    except Exception as e:  # pylint: disable=broad-except
      print('stopping............................')
      coord.request_stop(e)
    print('i should stop...')
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    #pickle.dump([out_dump, idx_dump, fn_dump], open('out.pkl', 'wb'))

if __name__ == '__main__':
  tf.app.run()
  print('DONE')
