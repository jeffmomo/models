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


slim = tf.contrib.slim

_TOTAL_CLASSES = 19027
_LEAF_DEPTH = 6

tf.app.flags.DEFINE_integer('current_level', _LEAF_DEPTH,
                            'The current level to train on, with 0 being the root')
tf.app.flags.DEFINE_boolean('native_indices', True,
                            'Whether to use the indices as specified by labels.txt')

current_level = tf.app.flags.FLAGS.current_level
native_indices = tf.app.flags.FLAGS.native_indices


hierarchy_tree = None




def_lst, idx_map = (0, 0)


class SpeciesDataset(object):

  _FILE_PATTERN = 'species_full_%s_*'

  SPLITS_TO_SIZES = {'train': 1000000, 'validation': 214141}


  _ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
    'filename': 'File name of the image',
  }

  class ModdedTensor(slim.tfexample_decoder.Tensor):

    def __init__(self, dataset, tensor_key, shape_keys=None, shape=None, default_value=0, use_native_indices=True,
                 translate=True):
      print(use_native_indices, translate)
      super(SpeciesDataset.ModdedTensor, self).__init__(tensor_key, shape_keys, shape, default_value)

      self.old_tensors_to_item = self.tensors_to_item
      self.tensors_to_item = self.new_tensors_to_item

      self.dataset = dataset
      self.translate = translate
      self.use_native_indices = use_native_indices or translate

    def new_tensors_to_item(self, keys_to_tensors):

      originals = self.old_tensors_to_item(keys_to_tensors)

      if not self.use_native_indices:
        indices = tf.squeeze(tf.gather(self.dataset.get_label_map(), originals))
        print('not using natives')
      else:
        indices = originals

      if self.translate:
        out = tf.gather(self.dataset.get_translation_map(), indices)
      else:
        out = indices

      return out

  def __init__(self, level=_LEAF_DEPTH):

    self.level = level
    self._NUM_CLASSES = _TOTAL_CLASSES
    self.translationmap_tensor = None
    self.labelmap_tensor = None

  def get_label_map(self):

    if self.labelmap_tensor is None:
      label_map = get_hierarchy.native_to_hierarchical_translation_map(hierarchy_tree)

      self.labelmap_tensor = tf.convert_to_tensor(label_map, tf.int64)

    return self.labelmap_tensor

  def get_translation_map(self):

    if self.translationmap_tensor is None:
      label_map = []

      for i in range(0, _TOTAL_CLASSES):
        label_map.append(get_hierarchy.translate_to_level(idx_map, self.level, i))

      self.translationmap_tensor = tf.convert_to_tensor(label_map, tf.int64)

    return self.translationmap_tensor

  def get_split(self, split_name, dataset_dir, file_pattern=None, reader=None):
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
    if split_name not in SpeciesDataset.SPLITS_TO_SIZES:
      raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
      file_pattern = SpeciesDataset._FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
      reader = tf.TFRecordReader


    # label_embeddings_batch = tf.gather(get_embeddings(), label_index_batch)


    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': SpeciesDataset.ModdedTensor(self, 'image/class/label', use_native_indices=native_indices, translate=not native_indices),
        'original_label': SpeciesDataset.ModdedTensor(self, 'image/class/label', use_native_indices=native_indices, translate=False),
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
        num_samples=SpeciesDataset.SPLITS_TO_SIZES[split_name],
        items_to_descriptions=SpeciesDataset._ITEMS_TO_DESCRIPTIONS,
        num_classes=self._NUM_CLASSES,
        labels_to_names=labels_to_names)


get_split = SpeciesDataset(current_level).get_split
