import subprocess
import argparse
import time
import hogwild
import redis
import traceback

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
        "--job_name",
        type=str,
        required=True,
        help="One of 'ps', 'worker'"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default='',
        help="Unique name for the experiment"
    )
    return parser 

def parse_file(lines):
    ps_hosts = []
    worker_hosts = []
    for l in lines:
        task_id, job, host_name = l.strip().split(',')
        if job == 'worker':
            worker_hosts.append('%s:%d' % (host_name, int(task_id) + 2223))
        if job == 'ps':
            ps_hosts.append('%s:%d' % (host_name, int(task_id) + 2222))
    return ps_hosts, worker_hosts

def main():
    parser = setup_parser()
    FLAGS, unparsed = parser.parse_known_args()
    host = subprocess.check_output(["hostname"]).strip()
    job_name = FLAGS.job_name
    if not job_name in ['ps', 'worker']:
        raise ValueError("Job name must be one of 'ps' or 'worker'")

    observed_workers = 0
    r = redis.StrictRedis(host=REDIS_HOST, port=6379, db=0)
    uid = int(r.rpush(job_name, host)) - 1
    while True:
        n_workers = int(r.llen('worker'))
        n_ps = int(r.llen('ps'))
        n = n_workers + n_ps
        if (n_workers == FLAGS.n_workers) and (n_ps == FLAGS.n_ps):
            break
        elif n > (FLAGS.n_workers + FLAGS.n_ps):
            raise ValueError('Too many connections, exiting...')
        else:
            print('Found %d / %d workers and %d / %d parameter servers. Waiting...' % (n_workers, FLAGS.n_workers, n_ps, FLAGS.n_ps))
            time.sleep(0.5)
    
    workers = r.lrange('worker',0, -1)
    worker_hosts = ['%s:%d' % (hostname, i + 2223) for hostname, i in zip(workers, range(len(workers)))]
    ps = r.lrange('ps', 0, -1)
    ps_hosts = ['%s:%d' % (hostname, i + 2210) for hostname, i in zip(ps, range(len(workers)))]
    print(worker_hosts, ps_hosts, job_name, uid)
    try:
        hogwild.run(worker_hosts, ps_hosts, job_name, uid)
        r.lrem(job_name, 1, host) # cleanup
    except Exception:
        with open('%s_%s.errors' % (job_name, FLAGS.task_index), 'w') as f:
            f.write(traceback.format_exc())
        r.lrem(job_name, 1, host) # cleanup



if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
