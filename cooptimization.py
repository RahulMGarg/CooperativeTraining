from __future__ import print_function

import argparse
import sys
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

from coop_training_optimizer import CooperativeReplicasOptimizer

FLAGS = None
DEFAULT_BASE_DIRECTORY = '/tmp'

def placeholder_inputs(batch_size=100, name_scope=None):
    with tf.name_scope(name_scope, 'Data'):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            28, 28, 1))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 10))
        return images_placeholder, labels_placeholder

def build_model(images, num_classes=10, name_scope=None):
    with tf.name_scope(name_scope, 'ConvNet'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
            net = slim.conv2d(images, 64, [5, 5])
            net = slim.max_pool2d(net)
            net = slim.conv2d(net, 64, [5, 5])
            net = slim.max_pool2d(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 192)
            net = slim.fully_connected(net, num_classes, activation_fn=None)       
            return net

def flatten(var):
    return tf.reshape(var, [-1])

def join_parameters(vars):
    out = flatten(vars[0])
    for i in xrange(1, len(vars)):
        out = tf.concat(0, [out, flatten(vars[i])])
    return out

def sqr_dist(x, y):
    return tf.reduce_sum(tf.square(x-y))

def weighting(x,y,sig=1.):
      return tf.exp(-sqr_dist(x,y) / (sig))

def replicate_vars_on_ps(local_variables, ps_device="/job:ps/task:0", trainable=False):
    ps_variables = []
    with tf.device(ps_device):
      for v in local_variables:
          ps_variables.append(tf.Variable(tf.zeros(v.get_shape()), trainable=trainable))
    ps_init = copy_variables(ps_variables, local_variables)
    return ps_variables, ps_init

def copy_variables(refs, values):
    copy_ops = []
    for r,v in zip(refs, values):
        copy_ops.append(tf.assign(r,v))
    return copy_ops

def get_batch():
    return batch_x, batch_y

def main(_):
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data, one_hot=True)
  if FLAGS.ps_hosts == "":
    with open('cooptimization.txt') as f:
        ps_hosts = [f.readlines()[0].strip() + ':2222']
  else:
    ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
      # Build model and set up local variables
      images, labels = placeholder_inputs(name_scope='worker_%d' % FLAGS.task_index)
      logits = build_model(images, name_scope='worker_%d' % FLAGS.task_index)
      local_variables = tf.trainable_variables()

    with tf.device("/job:ps/task:0"):
      global_step = tf.contrib.framework.get_or_create_global_step()
    
    # set up parameter server variables
    ps_variables, ps_init = replicate_vars_on_ps(local_variables, "/job:ps/task:0")
    
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
      # build optimization procedure
      slim.losses.softmax_cross_entropy(logits, labels)
      loss = slim.losses.get_total_loss()

      tf.summary.scalar('loss', loss)

      opt = tf.train.AdagradOptimizer(0.01)

      get_ps_state = copy_variables(local_variables, ps_variables)
      grads_and_vars = opt.compute_gradients(loss, local_variables)
      lr_offset = weighting(join_parameters(local_variables), join_parameters(ps_variables), sig=(1./FLAGS.sharpness))

      weighted_grads_and_vars = []
      for (g_loc, v_loc), v_ps in zip(grads_and_vars, ps_variables):
        weighted_grads_and_vars.append((g_loc * lr_offset, v_ps))
      train_op = opt.apply_gradients(weighted_grads_and_vars, global_step)

      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver(sharded=True)
    is_chief = FLAGS.task_index == 0
    merged_summaries = tf.summary.merge_all()
    global_init = tf.global_variables_initializer()

    with tf.Session(server.target) as sess:
      i = 0
      print('Starting training')
      if is_chief:
        # Copy the chief's initialization to the parameter server
        sess.run(global_init)
        sess.run(ps_init)
        train_writer = tf.summary.FileWriter(FLAGS.log_location + FLAGS.experiment,
                                      sess.graph)

      while True:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # sess.run handles AbortedError in case of preempted PS.
        sess.run(get_ps_state)

        if FLAGS.slow:
            pause = np.random.rand() * 2
            print("Sleeping for %f seconds." % pause, end=' ')
            time.sleep(pause)

        batch_x, batch_y = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        feed_dict = {images: batch_x.reshape((-1,28,28,1)), labels: batch_y}
        _, cost, step, summary, lr_weight = sess.run([train_op, loss, global_step, merged_summaries, lr_offset], 
                                    feed_dict=feed_dict)
        if is_chief:
            train_writer.add_summary(summary, global_step=step)

        print('Step: %d, cost: %f, lr_weight: %f' % (step, cost, lr_weight))
        if is_chief and step % 1000 == 0:
            saver.save(sess, FLAGS.log_location + FLAGS.experiment + '/logs')

        if step == FLAGS.max_steps:
            break

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )

  parser.add_argument(
      "--log_location",
      type=str,
      default= DEFAULT_BASE_DIRECTORY + "/distributed_training/",
      help="Directory for log storage"
  )

  parser.add_argument(
      "--experiment",
      type=str,
      default="/cooptimization/",
      help="Experiment name"
  )

  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )

  # Flags for training settings
  parser.add_argument(
      "--batch_size",
      type=int,
      default=100,
      help="Batch size.  Must divide evenly into the dataset sizes."
  )

  parser.add_argument(
      "--input_data_dir",
      type=str,
      default=DEFAULT_BASE_DIRECTORY + '/tensorflow/mnist/input_data',
      help="Directory to put the input data."
  )

  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000,
      help='Number of steps to run trainer.'
  )

  parser.add_argument(
      '--slow',
      action='store_true',
      help='Slow down training.'
  )

  parser.add_argument(
      "--sharpness",
      type=float,
      default=2.,
      help="How sharply we penalize incorrect parameter predictions."
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)