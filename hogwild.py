from __future__ import print_function

import argparse
import sys
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

from distributed_tf import placeholder_inputs, build_model

FLAGS = None

def main(_):
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data, one_hot=True)
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
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      # Build model and set up local variables
      images, labels = placeholder_inputs(name_scope='worker_%d' % FLAGS.task_index)
      logits = build_model(images, name_scope='worker_%d' % FLAGS.task_index)
      global_step = tf.contrib.framework.get_or_create_global_step()

      # build optimization procedure
      slim.losses.softmax_cross_entropy(logits, labels)
      loss = slim.losses.get_total_loss()

      tf.summary.scalar('loss', loss)

      opt = tf.train.AdagradOptimizer(0.01)

      train_op = opt.minimize(loss, global_step=global_step)

    is_chief = FLAGS.task_index == 0
    merged_summaries = tf.summary.merge_all()
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=FLAGS.log_location + FLAGS.experiment + '/logs',
                                           hooks=hooks) as sess:
      print('Starting training')
      if is_chief:
          train_writer = tf.summary.FileWriter(FLAGS.log_location + FLAGS.experiment,
                                      sess.graph)
      while not sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # sess.run handles AbortedError in case of preempted PS.

        batch_x, batch_y = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        feed_dict = {images: batch_x.reshape((-1,28,28,1)), labels: batch_y}
        _, cost, step, summary = sess.run([train_op, loss, global_step, merged_summaries], 
                                    feed_dict=feed_dict)
        if is_chief:
            train_writer.add_summary(summary, global_step=step)

        print('Step: %d, cost: %f' % (step, cost))

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
      default="/tmp/distributed_training/",
      help="Directory for log storage"
  )

  parser.add_argument(
      "--experiment",
      type=str,
      default="/hogwild/",
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
      default='/tmp/tensorflow/mnist/input_data',
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
      default=100000,
      help='Number of steps to run trainer.'
  )

  parser.add_argument(
      '--slow',
      action='store_true',
      help='Slow down training.'
  )


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)