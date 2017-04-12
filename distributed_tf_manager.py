import argparse
import subprocess

from jobfile_builder import populate_template, write_pbs_file

def build_jobstring(log_name='TEST', sync='hogwild', opt='sgd', predict_future=False, sharpness=20.):
    jobstring_template = "python torque_manager.py --log_name {log_name} --sync {sync} --opt {opt} --sharpness {sharpness}"
    if predict_future: 
        jobstring_template += ' --predict_future'
    js = jobstring_template.format(log_name=log_name,sync=sync, opt=opt, sharpness=sharpness) 
    js += " --task_index {task_index} --n_workers {n_workers}"
    return js

def build_joblist(n_workers, log_name='TEST', sync='hogwild', opt='sgd', predict_future=False, sharpness=20.):
    jobstring = build_jobstring(log_name=log_name, sync=sync, opt=opt, predict_future=predict_future, sharpness=sharpness)
    joblist = [(jobstring + ' --job_name ps').format(task_index=0, n_workers=n_workers)]
    joblist += [(jobstring + ' --job_name worker').format(task_index=i, n_workers=n_workers) for i in range(n_workers)]
    return joblist

def launch_job(filename, server='vickrey'):
    command = "echo ssh {server} 'qsub {filename}'".format(server=server, filename=filename)
    print command
    output = subprocess.check_output([command], shell=True).strip()
    print(output)

def main():
    args = parse_args()

    joblist = build_joblist(args.n_workers, log_name=args.name, sync=args.sync, opt=args.opt,
                            predict_future=args.predict_future, sharpness=args.sharpness)
    pbs_text = populate_template(name=args.name, joblist=joblist, gpu=args.gpu, 
                            theano=False, zero_index=True)
    write_pbs_file(args.name + '.pbs', pbs_text)
    if args.launch:
        launch_job(args.name + '.pbs')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the job immediately"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name for logging purposes"
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        required=True,
        help="Number of workers to use."
    )

    parser.add_argument(
        "--sync",
        type=str,
        default="hogwild",
        help="Method for syncing optimization algorithms"
    )

    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        help="Underlying optimization algorithm. Currently only support SGD and Adam." #TODO: add support for generic optimizers
    )

    ## Generic. TODO: abstract away from this...
    parser.add_argument(
        "--predict_future",
        action="store_true",
        default=False,
        help="Predict future states."
    )

    parser.add_argument(
        "--sharpness",
        type=float,
        default=20.,
        help="How sharply we penalize incorrect parameter predictions."
    )

    ### TORQUE OPTIONS

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Job uses a GPU"
    )

    parser.add_argument(
        "--memory",
        type=int,
        default="2000",
        help="Amount of memory in mb to request"
    )

    parser.add_argument(
        "--walltime",
        type=str,
        default="36:00:00",
        help="Amount of walltime to request"
    )
    return parser.parse_args()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
