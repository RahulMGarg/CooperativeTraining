from __future__ import print_function

import argparse
import sys
import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from cooptimization import placeholder_inputs, build_model, input_data, DEFAULT_BASE_DIRECTORY

FLAGS = None

DEFAULT_BASE_DIRECTORY = "/ubc/cs/project/arrow/jasonhar/CooperativeTraining"
LOG_LOCATION = DEFAULT_BASE_DIRECTORY + "/distributed_training/"
DATA_DIR = DEFAULT_BASE_DIRECTORY + '/tensorflow/mnist/input_data'
EXPERIMENT = "/hogwild/"
BATCH_SIZE = 100
MAX_STEPS = 1000000

def _p(s, p):
    if len(p) > 0:
        s = '%s_%s' % (p, s)
    return s

def run(worker_hosts, ps_hosts, job_name, task_index, logname="hogwild", opt="sgd", lr=0.01):
  EXPERIMENT = "/%s/" % logname
  settings = locals()
  data_sets = input_data.read_data_sets(DATA_DIR, one_hot=True)
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  log_file = _p('%s_%d.txt' % (job_name, task_index), logname)
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == "ps":
    with open(log_file, 'w') as f:
        f.write('Starting PS with settings: %s\n' % str(settings))
    server.join()
  elif job_name == "worker":
    with open(log_file, 'w') as f:
        f.write('Starting worker with settings: %s\n' % str(settings))

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):
      # Build model and set up local variables
      images, labels = placeholder_inputs(name_scope='worker_%d' % task_index)
      logits = build_model(images, name_scope='worker_%d' % task_index)
      global_step = tf.contrib.framework.get_or_create_global_step()

      # build optimization procedure
      slim.losses.softmax_cross_entropy(logits, labels)
      loss = slim.losses.get_total_loss()

      tf.summary.scalar('loss', loss)

      if opt == 'adam':
          opt = tf.train.AdamOptimizer(lr)
      elif opt == 'sgd':
          opt = tf.train.GradientDescentOptimizer(lr)
      else:
          raise ValueError('Unrecognised optimizer %s' % opt)

      train_op = opt.minimize(loss, global_step=global_step)

      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.summary.scalar('accuracy', accuracy)

    is_chief = task_index == 0
    merged_summaries = tf.summary.merge_all()

    global_init = tf.global_variables_initializer()
    sv = tf.train.Supervisor(logdir=LOG_LOCATION + EXPERIMENT,
                    is_chief=is_chief, init_op=global_init,
                    summary_op=None, checkpoint_basename='%s.ckpt' % logname)

    with open(log_file, 'a') as f:
      f.write('Ready to train\n')
    with sv.managed_session(server.target) as sess:
      print('Starting training')
      with open(log_file, 'a') as f:
        f.write('Starting training\n')
      while not sv.should_stop():
        batch_x, batch_y = data_sets.train.next_batch(BATCH_SIZE, False)
        feed_dict = {images: batch_x.reshape((-1,28,28,1)), labels: batch_y}
        _, cost, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
        if is_chief and step % 1 == 0:
            sv.summary_computed(sess, sess.run(merged_summaries, feed_dict=feed_dict))
        
        log_message = 'Step: %d, cost: %f' % (step, cost)
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')

        print(log_message)
        if step >= MAX_STEPS:
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
      default="/hogwild/",
      help="Experiment name"
  )

  parser.add_argument(
      "--opt",
      type=str,
      default="sgd",
      help="Optimizer"
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


  FLAGS, unparsed = parser.parse_known_args()
  DEFAULT_BASE_DIRECTORY = './'
  LOG_LOCATION = "./distributed_training/"
  DATA_DIR = './mnist/input_data'
  run(FLAGS.worker_hosts.split(','), FLAGS.ps_hosts.split(','), FLAGS.job_name, FLAGS.task_index, "LOCAL")
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
