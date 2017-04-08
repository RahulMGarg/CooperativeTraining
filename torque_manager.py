import subprocess
import argparse
import time
import hogwild

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

    observed_workers = 0
    with open('active_workers.txt', 'a') as f:
        f.write('%d,%s,%s\n' % (FLAGS.task_index, FLAGS.job_name, host))
    
    ps_ready = False
    workers_ready = False
    while True:
        with open('active_workers.txt') as f:
            lines = f.readlines()
        ps_hosts, worker_hosts = parse_file(lines)
        workers_ready = len(worker_hosts) == FLAGS.n_workers
        ps_ready = len(ps_hosts) == FLAGS.n_ps
        if workers_ready and ps_ready:
            break
        else:
            time.sleep(0.5)
    hogwild.run(worker_hosts, ps_hosts, FLAGS.job_name, FLAGS.task_index)
        



if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))