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
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.
Another difference is that 'v2' ResNets do not include an activation function in
the main pathway. Also see [2; Fig. 4e].

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

slim = tf.contrib.slim


def densenet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default DenseNet arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      # normalizer_fn=slim.batch_norm,
      # normalizer_params=batch_norm_params
  ):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def densenet(inputs,
              num_classes=None,
              is_training=True,
              growth_rate=12,
              drop_rate=0,
              depth=40,
              reuse=None,
              scope=None):
    """Generator for v2 (preactivation) ResNet models.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """

    n_channels = 2 * growth_rate
    reduction = 0.5
    bottleneck = True
    N = (depth - 4) / (6 if bottleneck else 3)

    """

    local function createModel(opt)

        --In our paper, a DenseNet-BC uses compression rate 0.5 with bottleneck structures
        --a default DenseNet uses compression rate 1 without bottleneck structures

        --N: #transformations in each denseblock
        local N = (opt.depth - 4)/3
        if bottleneck then N = N/2 end

        --non-bottleneck transformation
        local function addSingleLayer(model, nChannels, nOutChannels, dropRate)
          concate = nn.Concat(2)
          concate:add(nn.Identity())

          convFactory = nn.Sequential()
          convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
          convFactory:add(cudnn.ReLU(true))
          convFactory:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
          if dropRate>0 then
            convFactory:add(nn.Dropout(dropRate))
          end
          concate:add(convFactory)
          model:add(concate)
        end
        """

    def single_layer(input, n_out_channels, drop_rate):

        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, n_out_channels, [3, 3], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        out = tf.concat(1, [tf.identity(input), conv])

        return out
    """


        --bottleneck transformation
        local function addBottleneckLayer(model, nChannels, nOutChannels, dropRate)
          concate = nn.Concat(2)
          concate:add(nn.Identity())

          local interChannels = 4 * nOutChannels

          convFactory = nn.Sequential()
          convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
          convFactory:add(cudnn.ReLU(true))
          convFactory:add(cudnn.SpatialConvolution(nChannels, interChannels, 1, 1, 1, 1, 0, 0))
          if dropRate>0 then
            convFactory:add(nn.Dropout(dropRate))
          end

          convFactory:add(cudnn.SpatialBatchNormalization(interChannels))
          convFactory:add(cudnn.ReLU(true))
          convFactory:add(cudnn.SpatialConvolution(interChannels, nOutChannels, 3, 3, 1, 1, 1, 1))
          if dropRate>0 then
            convFactory:add(nn.Dropout(dropRate))
          end

          concate:add(convFactory)
          model:add(concate)
        end

        """
    def bottleneck_layer(input, n_output_channels, drop_rate):

        inter_channels = 4 * n_output_channels

        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, inter_channels, [1, 1], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        conv = slim.batch_norm(conv, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, inter_channels, [3, 3], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        out = tf.concat(1, [tf.identity(input), conv])
        return out

    """

        if bottleneck then
          add = addBottleneckLayer
        else
          add = addSingleLayer
        end

        local function addTransition(model, nChannels, nOutChannels, dropRate)
          model:add(cudnn.SpatialBatchNormalization(nChannels))
          model:add(cudnn.ReLU(true))
          model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
          if dropRate>0 then
            model:add(nn.Dropout(dropRate))
          end
          model:add(cudnn.SpatialAveragePooling(2, 2))
        end

        """

    if bottleneck:
        add = bottleneck_layer
    else:
        add = single_layer

    def transition(input, n_output_channels, drop_rate):
        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, n_output_channels, [1, 1], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)
        conv = slim.avg_pool2d(conv, [2, 2], stride=1)

        return conv

    with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_points_collection, padding='SAME'):
          with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs

            net = slim.conv2d(net, n_channels, [3, 3], stride=[1, 1])

            for i in range(0, N):
                net = add(net, growth_rate, drop_rate)
                n_channels += growth_rate

            net = transition(net, math.floor(n_channels * reduction), drop_rate)
            n_channels = math.floor(n_channels * reduction)

            for i in range(0, N):
                net = add(net, growth_rate, drop_rate)
                n_channels += growth_rate

            net = transition(net, math.floor(n_channels * reduction), drop_rate)
            n_channels = math.floor(n_channels * reduction)

            for i in range(0, N):
                net = add(net, growth_rate, drop_rate)
                n_channels += growth_rate

            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            net = slim.avg_pool2d(net, [8, 8], stride=[1, 1])


            if num_classes is not None:
              net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
              end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points

densenet.default_image_size = 224

"""

    model = nn.Sequential()


    --first conv before any dense blocks
    model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    --1st dense block and transition
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --2nd dense block and transition
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --3rd dense block
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end

    --global average pooling and classifier
    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    if opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
    elseif opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
    end


    --Initialization following ResNet
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
    end

    ConvInit('cudnn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end


    model:cuda()
    print(model)

    return model
end

return createModel
"""