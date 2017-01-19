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

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import eval_image_classifier

from hierarchy import get_hierarchy

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

hierarchy_tree = get_hierarchy.generate_tree()
hierarchy_tree.prune(threshold=100)

defs_lsts, idx_map = hierarchy_tree.get_tree_index_mappings()
definitions_list = defs_lsts[FLAGS.current_level]
num_classes = len(definitions_list)

translationmap_tensor = None

def get_translation_map():

    global translationmap_tensor

    if translationmap_tensor is None:
        translation_map = idx_map

        translationmap_tensor = tf.convert_to_tensor(translation_map[FLAGS.current_level], tf.int32)
        return translationmap_tensor

    return translationmap_tensor

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
    [image, label, filename, orig_label] = provider.get(['image', 'label', 'filename', 'original_label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels, filenames, original_labels = tf.train.batch(
        [image, label, filename, orig_label],
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
    labels = tf.squeeze(labels)

    # # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #     'Recall@5': slim.metrics.streaming_recall_at_k(
    #         logits, labels, 5),
    # })
    def write_out(maxes_out, labels_out, filenames_out, orig_labels_out):
      for idx in range(0, len(maxes_out)):
        # print(len(logits_out[idx]))

        assert labels_out[idx] >= 0 and labels_out[idx] < 2062, 'bad index..'

        print(maxes_out[idx], labels_out[idx], orig_labels_out[idx])

        print("classified:", definitions_list[maxes_out[idx]], "ident:", definitions_list[labels_out[idx]], "original ident:", defs_lsts[-1][orig_labels_out[idx]])

        # do some stuff to add back the background class expected by my processing functions
        # print(">", ','.join(['-9.99999'] + ['%.5f' % num for num in logits_out[idx]]))
        # print(">ident", (labels_out[idx] + 1), filenames_out[idx])

      return maxes_out, labels_out, filenames_out, orig_labels_out

    eval_op = [tf.nn.in_top_k(logits, labels, 1), tf.nn.in_top_k(logits, labels, 5), tf.py_func(write_out, [predictions, labels, filenames, original_labels], [tf.int64, tf.int32, tf.string, tf.int32])]

    final_op = [tf.identity(logits), tf.identity(labels), tf.identity(filenames)]

    # # Print the summaries to screen.
    # for name, value in names_to_values.iteritems():
    #   print(name, value)
    #   summary_name = 'eval/%s' % name
    #   op = tf.scalar_summary(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

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

    # print(logits_out)
    # for idx in xrange(0, len(logits_out)):
    #   print(len(logits_out[idx]))
    #   # do some stuff to add back the background class expected by my processing functions
    #   print(">", ','.join(['-99.99999'] + ['%.5f' % num for num in logits_out[idx]]))
    #   print(">ident", (labels_out[idx] + 1), filenames_out[idx])


if __name__ == '__main__':
  tf.app.run()