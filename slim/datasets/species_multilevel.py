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
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/data/create_cifar10_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

from hierarchy import get_hierarchy

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

SPLITS_TO_SIZES = {'train': 663832, 'validation': 168607}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
    'original_label': 'The original label',
    'filename': 'File name of the image',
}

tf.app.flags.DEFINE_integer('current_level', 0,
                            'The current level to train on, with 0 being the root')

current_level = tf.app.flags.FLAGS.current_level

hierarchy_tree = get_hierarchy.generate_tree()
hierarchy_tree.prune(threshold=100)

assert len(hierarchy_tree.children()) == 2062, "Unexpected number of classes"

def_lst, idx_map = hierarchy_tree.get_tree_index_mappings()

_TOTAL_CLASSES = 2062
_NUM_CLASSES = len(def_lst[current_level])

labelmap_tensor = None

def get_label_map():

    if labelmap_tensor is None:

        label_map = []
        file = open('/home/dm116/Workspace/MultiLevelSoftmax/index_map.dat', 'r')
        for line in file:
            label_map.append(int(line[:-1]))

        global labelmap_tensor
        labelmap_tensor = tf.convert_to_tensor(label_map, tf.int32)
        return labelmap_tensor

    return labelmap_tensor


translationmap_tensor = None

def get_translation_map():

    global translationmap_tensor

    if translationmap_tensor is None:
        label_map = []

        for i in range(0, _TOTAL_CLASSES):
            label_map.append(get_hierarchy.translate_to_level(idx_map, current_level, i))


        translationmap_tensor = tf.convert_to_tensor(label_map, tf.int32)
        return translationmap_tensor

    return translationmap_tensor


embeddings_tensor = None

def get_embeddings_map():
    global embeddings_tensor

    if embeddings_tensor is None:

        embeddings_tensor = tf.convert_to_tensor(idx_map, tf.float32)

        return embeddings_tensor
    else:
        return embeddings_tensor



class ModdedTensor(slim.tfexample_decoder.Tensor):

  def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0, translate=True):
    super(ModdedTensor, self).__init__(tensor_key, shape_keys, shape, default_value)

    self.old_tensors_to_item = self.tensors_to_item
    self.tensors_to_item = self.new_tensors_to_item

    self.translate = translate


  def new_tensors_to_item(self, keys_to_tensors):

    if self.translate:
      out = tf.gather(get_translation_map(), tf.squeeze(tf.gather(get_label_map(), self.old_tensors_to_item(keys_to_tensors))) - 1)
    else:
      out = tf.squeeze(tf.gather(get_label_map(), self.old_tensors_to_item(keys_to_tensors))) - 1

    return out


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader


  # label_embeddings_batch = tf.gather(get_embeddings(), label_index_batch)


  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=-1),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': ModdedTensor('image/class/label', translate=True),
      'original_label': ModdedTensor('image/class/label', translate=False),
      'filename': slim.tfexample_decoder.Tensor('image/filename'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
