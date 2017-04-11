from __future__ import print_function

import argparse
import sys
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None
DEFAULT_BASE_DIRECTORY = '/tmp'

class OnlineUpdate(object):
    def __init__(self, grads_and_vars):
        self.variables = [v for g,v in grads_and_vars]
        self.grads = [g for g,v in grads_and_vars]
        
        with tf.name_scope('OnlinePrediction'):
            self.locals = []
            self.prev_prediction = []
            for v in self.variables:
                W_p = tf.Variable(tf.ones_like(v.initialized_value()), trainable=False, name='W_parameter')
                W_g = tf.Variable(tf.zeros_like(v.initialized_value()), trainable=False, name='W_grad')
                bias = tf.Variable(tf.zeros_like(v.initialized_value()), trainable=False, name='bias')
                p = tf.Variable(tf.zeros_like(v.initialized_value()), trainable=False, name='prediction')
                self.prev_prediction.append(p)
                self.locals.append((W_p, W_g, bias))
    
    def get_locals(self):
        '''
        Get all local variables that need to be initialized
        '''
        local_vars = self.prev_prediction
        for pars in self.locals:
            for v in pars:
                local_vars.append(v)
        return local_vars

    def get_predict_op(self):
        predict_op = []
        for v, g, w, p in zip(self.variables, self.grads, self.locals, self.prev_prediction):
            prediction = v * w[0] + g * w[1] + w[2]
            predict_op.append(tf.assign(v, prediction))
            predict_op.append(tf.assign(p, prediction))
        return predict_op
    
    def update_variables(self, true_values, lr = 0.0001, update='sgd', eps = 1e-6):
        update_op = []
        if update == 'adagrad':
            h_list = []
        for y, v, g, w in zip(true_values, self.prev_prediction, self.grads, self.locals):
            y_hat = v * w[0] + g * w[1] + w[2]
            diff = (y - y_hat)
            d_yhat = [(w[0], v), (w[1],g), (w[2], 1)]
            for weight, partial in d_yhat:
                gradient = -2 * diff * partial
                if update == 'sgd':
                    update_op.append(tf.assign(weight, weight - lr * gradient))
                elif update == 'adagrad':
                    with tf.name_scope('OnlineAdagrad'):
                        h = tf.Variable(tf.zeros_like(weight), trainable=False, name=weight)
                    h_list.append(h)
                    update_op.append(tf.assign(h, h + tf.square(gradient)))
                    gradient = gradient / (eps + tf.sqrt(h))
                    update_op.append(tf.assign(weight, weight - lr * gradient))
        return update_op


def placeholder_inputs(batch_size=100, name_scope=None):
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

def select_optimizer(opt):
    if opt == 'adam':
        return tf.train.AdamOptimizer(0.01)
    elif opt == 'sgd':
        return tf.train.GradientDescentOptimizer(0.01)
    else:
        raise ValueError('Unrecognised optimizer %s' % opt)

def flatten(var):
    return tf.reshape(var, [-1])

def join_parameters(vars):
    out = flatten(vars[0])
    for i in xrange(1, len(vars)):
        out = tf.concat([out, flatten(vars[i])], 0)
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

def _p(s, p):
    if len(p) > 0:
        s = '%s_%s' % (p, s)
    return s

DEFAULT_BASE_DIRECTORY = "/ubc/cs/project/arrow/jasonhar/CooperativeTraining"
LOG_LOCATION = DEFAULT_BASE_DIRECTORY + "/distributed_training/"
DATA_DIR = DEFAULT_BASE_DIRECTORY + '/tensorflow/mnist/input_data'
EXPERIMENT = "/hogwild/"
BATCH_SIZE = 100
MAX_STEPS = 1000000

def run(worker_hosts, ps_hosts, job_name, task_index,
        logname="cooptimzation", opt="adam", predict_future=False, sharpness=2.):
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
    with tf.device("/job:worker/task:%d" % task_index):
      # Build model and set up local variables on the worker nodes
      images, labels = placeholder_inputs(name_scope='worker_%d' % task_index)
      logits = build_model(images, name_scope='worker_%d' % task_index)
      local_variables = tf.trainable_variables()

    with tf.device("/job:ps/task:0"):
      global_step = tf.contrib.framework.get_or_create_global_step()
    
    # set up parameter server variables
    ps_variables, ps_init = replicate_vars_on_ps(local_variables, "/job:ps/task:0")

    with tf.device("/job:worker/task:%d" % task_index):
      # build optimization procedure
      slim.losses.softmax_cross_entropy(logits, labels)
      loss = slim.losses.get_total_loss()

      tf.summary.scalar('loss', loss)

      opt = select_optimizer(opt)

      get_ps_state = copy_variables(local_variables, ps_variables)
      grads_and_vars = opt.compute_gradients(loss, local_variables)
      additional_variables = [] # placeholder to manage inits - only used if we use the prediction operation
      if predict_future:
        print('Building prediction operation')
        online_update = OnlineUpdate(grads_and_vars)
        predict_op = online_update.get_predict_op()
        update_op = online_update.update_variables(ps_variables)
        additional_variables = online_update.get_locals()
      
      local_join = join_parameters(local_variables)
      ps_join = join_parameters(ps_variables)
      lr_offset = weighting(local_join, ps_join, sig=(1./sharpness))

      weighted_grads_and_vars = []
      for (g_loc, v_loc), v_ps in zip(grads_and_vars, ps_variables):
        weighted_grads_and_vars.append((g_loc * lr_offset, v_ps))
      train_op = opt.apply_gradients(weighted_grads_and_vars, global_step)

      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.summary.scalar('accuracy', accuracy)

    is_chief = task_index == 0
    merged_summaries = tf.summary.merge_all()

    #other = tf.global_variables()
    #print(other)
    local_init = tf.variables_initializer(local_variables + additional_variables)

    sv = tf.train.Supervisor(logdir=LOG_LOCATION + EXPERIMENT,
                    is_chief=is_chief,
                    summary_op=None,
                    saver=None, # remove this
                    checkpoint_basename='%s.ckpt' % logname,
                    local_init_op=local_init)

    with sv.managed_session(server.target) as sess:
      if is_chief:
        sess.run(ps_init)
      print('Starting training')
      with open(log_file, 'a') as f:
        f.write('Starting training\n')
      while not sv.should_stop():
        batch_x, batch_y = data_sets.train.next_batch(BATCH_SIZE, False)
        feed_dict = {images: batch_x.reshape((-1,28,28,1)), labels: batch_y}
        sess.run(get_ps_state)
        if predict_future:
            _ = sess.run(predict_op, feed_dict=feed_dict)

        if predict_future:
            _, cost, step, lr_weight, _ = sess.run([train_op, loss, global_step, lr_offset, update_op], 
                                        feed_dict=feed_dict)
        else:
            _, cost, step, lr_weight = sess.run([train_op, loss, global_step, lr_offset], 
                                        feed_dict=feed_dict)

        if is_chief:
            sv.summary_computed(sess, sess.run(merged_summaries, feed_dict=feed_dict))

        log_message = 'Step: %d, cost: %f, lr_weight: %f' % (step, cost, lr_weight)
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')

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
      '--predict_future',
      default=False,
      help='Predict future states.',
      action='store_true'
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
  DEFAULT_BASE_DIRECTORY = './'
  LOG_LOCATION = "./distributed_training/"
  DATA_DIR = './mnist/input_data'
  run(FLAGS.worker_hosts.split(','), FLAGS.ps_hosts.split(','), FLAGS.job_name, FLAGS.task_index, "LOCAL", predict_future=FLAGS.predict_future)
