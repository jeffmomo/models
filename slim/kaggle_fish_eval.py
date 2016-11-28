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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import threading

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import eval_image_classifier

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

outfile = open('eval_out.csv', 'w+')
filename_set = set()
lock = threading.Lock()

def writeout(txt):
    outfile.write(txt)
    outfile.write('\n')
    outfile.flush()

writeout('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT')

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

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

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, filename] = provider.get(['image', 'filename'])

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, filenames = tf.train.batch(
        [image, filename],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

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

    softmax = tf.nn.softmax(logits)

    


    def write_out(logits_out, filenames_out):
      for idx in xrange(0, len(logits_out)):
        
        fname = filenames_out[idx].split('/')[-1]
        output = ','.join([fname] + ['%.15f' % num for num in logits_out[idx]])

        processed = int(fname.split('.')[0].split('_')[1])
        lock.acquire()
        if processed in filename_set:
            print('dup: ' + str(processed))
            continue
            # exit(-1)
        filename_set.add(processed)
        lock.release()

        writeout(output)

      return logits_out, filenames_out

    eval_op = [tf.py_func(write_out, [softmax, filenames], [tf.float32, tf.string])]


    # def final(softmax):
    #     print(filename_set)
    #     for i in xrange(0, 1000):
    #         if not i in filename_set:
    #             print('missing')

    #     return softmax


    # final_op = [tf.py_func(final, [softmax], [tf.float32])]


    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_op,
        # final_op=final_op,
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
