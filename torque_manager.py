import subprocess
import argparse
import time
import traceback

import redis

import hogwild
import cooptimization

REDIS_HOST = 'cersei'

def setup_parser():
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="The total number of workers for the job"
    )

    parser.add_argument(
        "--n_ps",
        type=int,
        default=1,
        help="The total number of parameter servers for the job"
    )

    parser.add_argument(
        "--log_name",
        type=str,
        required=True,
        help="Name to use for redis queue"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        required=True,
        help="One of 'ps', 'worker'"
    )

    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        help="Optimizer: sgd or adam currently"
    )

    parser.add_argument(
        "--sync",
        type=str,
        default="hogwild",
        help="Syncronization method - defaults to hogwild (i.e. no syncronization)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The initial learning rate for the optimizers"
    )

    parser.add_argument(
        "--sharpness",
        type=float,
        default=2.,
        help="The sharpness parameter for the soft gating function used in the cooptimization method"
    )

    parser.add_argument(
        "--predict_future",
        default=False,
        action="store_true",
        help="Flag to turning on prediction of future state for the cooptimization method"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default='',
        help="Unique name for the experiment"
    )
    return parser 

def decode_list(redis_list):
    return [eval(i) for i in redis_list]

def get_runner(sync_method, FLAGS):
    if sync_method == 'hogwild':
        return lambda worker_hosts, ps_hosts, job_name, task_index, log_name: hogwild.run(worker_hosts, ps_hosts, job_name, task_index, log_name, FLAGS.opt, lr=FLAGS.lr)
    elif sync_method == 'coop':
        return lambda worker_hosts, ps_hosts, job_name, task_index, log_name: cooptimization.run(worker_hosts, ps_hosts, job_name, task_index,
                                          log_name, FLAGS.opt, FLAGS.predict_future, FLAGS.sharpness, lr=FLAGS.lr)
    else:
        raise ValueError("Unrecognized synchronization method: %s" % sync_method)


def main():
    parser = setup_parser()
    FLAGS, unparsed = parser.parse_known_args()
    def job_id(job_name):
        return '%s::%s' % (FLAGS.log_name, job_name)
    host = subprocess.check_output(["hostname"]).strip()
    job_name = FLAGS.job_name
    if not job_name in ['ps', 'worker']:
        raise ValueError("Job name must be one of 'ps' or 'worker'")
    
    # Use redis queue to check that all workers are online
    observed_workers = 0
    r = redis.StrictRedis(host=REDIS_HOST, port=6379, db=0)
    unique_name = "('%s', %d)" % (host, FLAGS.task_index)
    uid = int(r.rpush(job_id(job_name), unique_name)) - 1
    while True:
        n_workers = int(r.llen(job_id('worker')))
        n_ps = int(r.llen(job_id('ps')))
        n = n_workers + n_ps
        if (n_workers == FLAGS.n_workers) and (n_ps == FLAGS.n_ps):
            break
        elif n > (FLAGS.n_workers + FLAGS.n_ps):
            raise ValueError('Too many connections, exiting...')
        else:
            print('Found %d / %d workers and %d / %d parameter servers. Waiting...' % (n_workers, FLAGS.n_workers, n_ps, FLAGS.n_ps))
            time.sleep(0.5)
    
    workers = decode_list(r.lrange(job_id('worker'),0, -1))
    worker_hosts = ['%s:%d' % (hostname, i + 2223) for hostname, i in workers]
    ps = decode_list(r.lrange(job_id('ps'), 0, -1))
    ps_hosts = ['%s:%d' % (hostname, i + 2210) for hostname, i in ps]
    print(worker_hosts, ps_hosts, job_id(job_name), FLAGS.task_index)
    run = get_runner(FLAGS.sync, FLAGS)
    try:
        run(worker_hosts, ps_hosts, job_name, FLAGS.task_index, FLAGS.log_name)
        r.lrem(job_id(job_name), 1, unique_name) # cleanup
    except Exception:
        with open('%s_%s.errors' % (job_id(job_name), FLAGS.task_index), 'w') as f:
            f.write(traceback.format_exc())
        r.lrem(job_id(job_name), 1, unique_name) # cleanup



if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
