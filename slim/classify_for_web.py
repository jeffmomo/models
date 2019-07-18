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

import threading
import base64
import math
import cv2
import numpy as np
import sys
import tensorflow as tf
import os

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

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
#----------------------------------------------

def setup_queue(sess):
    queue = tf.FIFOQueue(
        capacity=1000,
        dtypes=[tf.string, tf.int32])

    main_thread_graph = tf.get_default_graph()

    def runFeeder():
        with main_thread_graph.as_default():
            while True:
                f = open(os.path.expanduser('~/classify_pipe.fifopipe'), 'r')
                lines = f.readlines()
                entries = []
                # Each read of the pipe may contain more than 1 image.
                for line in lines:
                    # Here you parse whatever you have sent through the pipe.
                    # My admittingly horrible format is: image_bytes >>>DATA<<< image_index >>>EOF<<<
                    # Where image_bytes is the base64 encoded image, >>>DATA<<< and >>>EOF<<< are delimiters, and image_id is the image ID
                    entries.extend([(base64.standard_b64decode(print(x[0]) or x[0]), int(x[1].split('|')[0])) for x in
                               [x.split('>>>DATA<<<') for x in line.split(">>>EOF<<<")[:-1]]])

                for img, image_id in entries:
                    img_np = None

                    try:
                        img_np = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
                    except:
                        sys.stderr.write('bad image passed\n')

                    # The classifier expects jpeg format
                    conv_img_bytes = bytes(cv2.imencode('.jpeg', img_np)[1])

                    # Maybe have a look at what was sent thorugh?
                    cv2.startWindowThread()
                    cv2.namedWindow("preview")
                    cv2.imshow("Preview for image " + str(image_id), img_np)


                    if img_np == None:
                        # Uses a negative image_id to indicate failure
                        conv = (tf.convert_to_tensor(conv_img_bytes, tf.string), tf.convert_to_tensor(-int(image_id), tf.int32))
                    else:
                        conv = (tf.convert_to_tensor(conv_img_bytes, tf.string), tf.convert_to_tensor(image_id, tf.int32))

                    # enqueues a tuple of (Str, Int)
                    enqueue_op = queue.enqueue(vals=conv)
                    sess.run(enqueue_op)

    threading.Thread(target=runFeeder).start()
    return queue


def setup_queue_with_priors(sess):
    queue = tf.FIFOQueue(
        capacity=1000,
        dtypes=[tf.string, tf.int32, tf.string])

    main_thread_graph = tf.get_default_graph()

    def runFeeder():
        with main_thread_graph.as_default():
            while True:
                f = open(os.path.expanduser('~/classify_pipe.fifopipe'), 'r')
                lines = f.readlines()
                ls = []
                for line in lines:
                    print('lines lines lines')
                    ls.extend([(base64.standard_b64decode(x[0]), int(x[1].split('|')[0]), x[1].split('|')[1]) for x in
                               [x.split('>>>INDEX<<<') for x in line.split(">>>EOF<<<")[:-1]]])

                for img, idx, priors in ls:
                    img_np = None

                    try:
                        img_np = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
                    except:
                        sys.stderr.write('bad image passed\n')
                        continue

                    if img_np is None:
                        sys.stderr.write('bad image passed\n')
                        continue

                    conv_img_bytes = bytes(cv2.imencode('.jpeg', img_np)[1])

                    print('img recvd: ' + str(idx))
                    # cv2.startWindowThread()
                    # cv2.namedWindow("preview")
                    # cv2.imshow("preview", img_np)

                    # enqueues a (string, int, string)
                    conv = (tf.convert_to_tensor(conv_img_bytes, tf.string), tf.convert_to_tensor(idx, tf.int32),
                            tf.convert_to_tensor(priors, tf.string))

                    enqueue_op = queue.enqueue(vals=conv)
                    sess.run(enqueue_op)


                    # starts feed thread

    threading.Thread(target=runFeeder).start()

    return queue

def classify_input_single_with_priors(queue, preprocess_fn):

    img_bytes, image_id, priors = queue.dequeue()

    image = preprocess_fn(img_bytes, 0, False)

    height = width = FLAGS.image_size
    depth = 3

    image = tf.cast(image, tf.float32)
    # Reshape images into these desired dimensions.
    image = tf.reshape(image, shape=[1, height, width, depth])

    # Display the training images in the visualizer.
    # tf.image_summary('images', images)

    return image, image_id, priors  # tf.reshape(filename_batch, [1])

#----------------------------------------------


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    #------------------

    sess = tf.Session(graph=tf.get_default_graph())
    queue = setup_queue_with_priors(sess)

    #------------------

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    # DELET
    # ##############################################################
    # # Create a dataset provider that loads data from the dataset #
    # ##############################################################
    # provider = slim.dataset_data_provider.DatasetDataProvider(
    #     dataset,
    #     shuffle=False,
    #     common_queue_capacity=2 * FLAGS.batch_size,
    #     common_queue_min=FLAGS.batch_size)
    # [image, label] = provider.get(['image', 'label'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    # REPLACE THIS
    # image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    #
    # images, labels = tf.train.batch(
    #     [image, label],
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.num_preprocessing_threads,
    #     capacity=5 * FLAGS.batch_size)

    #WITH THIS
    #------------------
    img_bytes, image_id, _ = queue.dequeue()
    num_channels = 3

    image = tf.image.decode_jpeg(img_bytes, channels=num_channels)# preprocess_fn(img_bytes, 0, False)
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    # Reshape to be like minibatch
    images = tf.reshape(image, [1, eval_image_size, eval_image_size, num_channels])

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)

    #DELET
    # labels = tf.squeeze(labels)

    #
    # # TODO(sguada) use num_epochs=1
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    #--------------



    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)

    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        print("starting classification")

        while not coord.should_stop():
            print(sess.run([predictions]))


    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)





if __name__ == '__main__':
  tf.app.run()
