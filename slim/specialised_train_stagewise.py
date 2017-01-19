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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.contrib.layers import initializers
from tensorflow.core.protobuf import saver_pb2

from hierarchy import get_hierarchy

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

import datasets.species_multilevel_specialised as species_multilevel_specialised

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

import preprocessing.species_preprocessing as species_preprocessing

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')




################
# Custom Flags #
################

tf.app.flags.DEFINE_boolean(
    'expand_logits', False,
    'Expands the logits layer according to our current level (Flag is specified in species_multilevel.py)')

tf.app.flags.DEFINE_string('schedule', 'default_schedule',
                           'The training schedule to use - i.e. when to switch training levels')

FLAGS = tf.app.flags.FLAGS


hierarchy_tree = get_hierarchy.generate_tree()
hierarchy_tree.prune(threshold=100)

def_lst, idx_map = hierarchy_tree.get_tree_index_mappings()





def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.histogram_summary(variable.op.name, variable))
  summaries.append(tf.scalar_summary('training/Learning Rate', learning_rate))
  return summaries


class MyBuilder(BaseSaverBuilder):

  def __init__(self, level, expand=False):
    super(MyBuilder, self).__init__()

    self.expand = expand
    self.level = level

    self.translationmap_tensors = []

    for item in idx_map:
      self.translationmap_tensors.append(tf.convert_to_tensor(item, tf.int32))

  def restore_op(self, filename_tensor, saveable, preferred_shard):
    """Create ops to restore 'saveable'.
    Args:
      filename_tensor: String Tensor.
      saveable: A BaseSaverBuilder.SaveableObject object.
      preferred_shard: Int.  Shard to open first when loading a sharded file.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.
    """
    # pylint: disable=protected-access
    from tensorflow.python.ops import io_ops

    lsw = ["InceptionResnetV2/Logits/Logits/weights", "InceptionResnetV2/AuxLogits/Logits/weights"] if self.expand else []
    lsb = ["InceptionResnetV2/AuxLogits/Logits/biases", "InceptionResnetV2/Logits/Logits/biases"] if self.expand else []



    tensors = []
    for spec in saveable.specs:

      restored = io_ops._restore_slice(
            filename_tensor,
            spec.name,
            spec.slice_spec,
            spec.tensor.dtype,
            preferred_shard=preferred_shard)

      if spec.name in lsw:

        to_append = tf.transpose(tf.gather(tf.transpose(restored), self.translationmap_tensors[self.level]))
        print(to_append, to_append.get_shape())
        to_append = to_append + tf.random_uniform(tf.shape(to_append), minval=-0.001, maxval=0.001) #(initializers.xavier_initializer()([int(spec.slice_spec.split(' ')[0]), len(idx_map[self.level])]) / 3)
        # todo: Explore the effects of randomness, and what variance I should use.

        tensors.append(to_append)

      elif spec.name in lsb:

        to_append = tf.transpose(tf.gather(tf.transpose(restored), self.translationmap_tensors[self.level]))
        tensors.append(to_append)

      else:
        tensors.append(restored)

    return tensors

def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False, expand_logits=False, level=0):
  """Returns a function that assigns specific variables from a checkpoint.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
        use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
        checkpoint to the correspoing variables to initialize. If empty or None,
        it would return  no_op(), None.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
        the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
        which are of different shape then the ones stored in the checkpoint but
        which have the same number of elements.

  Returns:
    A function that takes a single argument, a `tf.Session`, that applies the
    assignment operation.

  Raises:
    ValueError: If the checkpoint specified at `model_path` is missing one of
      the variables in `var_list`.
  """

  if ignore_missing_vars:
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
      var_dict = var_list
    else:
      var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:
      if reader.has_tensor(var):
        available_vars[var] = var_dict[var]
      else:
        tf.logging.warning(
            'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
  saver = tf.train.Saver(var_list, reshape=reshape_variables, builder=MyBuilder(level, expand_logits))
  def callback(session):
    saver.restore(session, model_path)
  return callback

def _get_init_fn(level, checkpoint_path, expand, restore_logits):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if not restore_logits:
    exclusions = [scope.strip()
                  for scope in 'InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/Logits'.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)



  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  print(FLAGS.ignore_missing_vars, FLAGS.expand_logits)

  return assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars,
      expand_logits=expand,
      level=level)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train









def one_train_cycle(current_level, checkpoint_path, expand_logits, restore_logits=True):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')


  # translationmap_tensors = None
  # def get_translation_map():
  #
  #   global translationmap_tensors
  #
  #   if translationmap_tensors is None:
  #     print(idx_map)
  #     translationmap_tensors = []
  #
  #     for item in idx_map:
  #       translationmap_tensors.append(tf.convert_to_tensor(item, tf.int32))
  #
  #   return translationmap_tensors


  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():

    ######################
    # Config model_deploy#
    ######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = species_multilevel_specialised.SpeciesDataset(current_level).get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the network #
    ####################
    network_fn = nets_factory.get_network_fn(
        "inception_resnet_v2",
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    image_preprocessing_fn = species_preprocessing.preprocess_image

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size, is_training=True)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weight=0.4, scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weight=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.histogram_summary('activations/' + end_point, x))
      summaries.add(tf.scalar_summary('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.scalar_summary('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.histogram_summary(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.scalar_summary('learning_rate', learning_rate,
                                      name='learning_rate'))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.scalar_summary('total_loss', total_loss,
                                    name='total_loss'))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)

    print(list(map(lambda x: (x.op.name, x, tf.shape(x)), tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'InceptionResnetV2/Logits'))))



    train_tensor = control_flow_ops.with_dependencies([update_op], tf.Print(total_loss, tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'InceptionResnetV2/Logits')),
                                                      name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.merge_summary(list(summaries), name='summary_op')

    ###########################
    # Kicks off the training. #
    ###########################

    saver = tf.train.Saver()
    saver.old_save = saver.save
    save_path = None
    def new_save(*args, **kwargs):
      nonlocal save_path

      save_path = saver.old_save(*args, **kwargs)

    saver.save = new_save

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(current_level, checkpoint_path, expand_logits, restore_logits),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        saver=saver,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)

    print("doneee")
    print('saved at', save_path)

    return save_path



def basic_schedule(checkpoint_path, from_level=3, to_level=6, max_steps_per_level=10000):
  initial_train_dir = FLAGS.train_dir

  FLAGS.train_dir = initial_train_dir + str(from_level)
  FLAGS.max_number_of_steps = max_steps_per_level


  checkpoint_path = one_train_cycle(from_level, checkpoint_path, False, False)

  for i in range(from_level + 1, to_level + 1):
    FLAGS.train_dir = initial_train_dir + str(i)
    checkpoint_path = one_train_cycle(i, checkpoint_path, True)

def species_schedule(checkpoint_path):
  initial_train_dir = FLAGS.train_dir

  FLAGS.max_number_of_steps = 10000
  FLAGS.train_dir = initial_train_dir + '3'
  checkpoint_path = one_train_cycle(3, checkpoint_path, False, False)

  FLAGS.max_number_of_steps = 15000
  FLAGS.train_dir = initial_train_dir + '4'
  checkpoint_path = one_train_cycle(4, checkpoint_path, True)

  FLAGS.max_number_of_steps = 20000
  FLAGS.train_dir = initial_train_dir + '5'
  checkpoint_path = one_train_cycle(5, checkpoint_path, True)

  FLAGS.max_number_of_steps = 50000
  FLAGS.train_dir = initial_train_dir + '6'
  checkpoint_path = one_train_cycle(6, checkpoint_path, True)

def default_schedule(checkpoint_path):
  basic_schedule(checkpoint_path, 0, 6, 10)

def complete_train_schedule(checkpoint_path):
    initial_train_dir = FLAGS.train_dir

    FLAGS.learning_rate = 0.5
    FLAGS.max_number_of_steps = 5000
    FLAGS.train_dir = initial_train_dir + '2'
    checkpoint_path = one_train_cycle(2, checkpoint_path, False, False)

    FLAGS.learning_rate = 0.4
    FLAGS.max_number_of_steps = 5000
    FLAGS.train_dir = initial_train_dir + '3'
    checkpoint_path = one_train_cycle(3, checkpoint_path, True)

    FLAGS.learning_rate = 0.2
    FLAGS.max_number_of_steps = 10000
    FLAGS.train_dir = initial_train_dir + '4'
    checkpoint_path = one_train_cycle(4, checkpoint_path, True)

    FLAGS.learning_rate = 0.2
    FLAGS.max_number_of_steps = 10000
    FLAGS.train_dir = initial_train_dir + '5'
    checkpoint_path = one_train_cycle(5, checkpoint_path, True)

    FLAGS.learning_rate = 0.01
    FLAGS.max_number_of_steps = 20000
    FLAGS.train_dir = initial_train_dir + '6'
    checkpoint_path = one_train_cycle(6, checkpoint_path, True)

schedules = {
    'basic_schedule': basic_schedule,
    'default_schedule': default_schedule,
    'complete_train_schedule': complete_train_schedule,
}

def main(_):

  if FLAGS.checkpoint_path is None:
      checkpoint_path = None
  elif tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  schedules[FLAGS.schedule](checkpoint_path)

  # run_basic_schedule(checkpoint_path, 3, 6, 10)
  # initial_train_dir = FLAGS.train_dir
  #
  # FLAGS.train_dir = initial_train_dir + '0'
  # checkpoint_path = one_train_cycle(0, checkpoint_path, False)
  # FLAGS.train_dir = initial_train_dir + '1'
  # one_train_cycle(1, checkpoint_path, True)




if __name__ == '__main__':
  tf.app.run()
