import argparse
import re
from itertools import product
from jobfile_builder import populate_template, write_pbs_file

worker_jobstring = 'python {model}.py --worker_hosts={workerhosts} --job_name={jobname} --task_index={index}'
ps_jobstring = 'hostname > {model}_ps_host.txt; python {model}.py --worker_hosts={workerhosts} --job_name={jobname} --task_index={index}'
ada_nodes = ['ada{}:{}'.format(i, 2223+j) for i in range(17,22) for j in range(4)]

def main():
    experiment = 'hogwild'
    workerhosts = ada_nodes
    pshost = 'localhost'

    worker_pbs = populate_template(name=experiment, joblist=[worker_jobstring.format(model=experiment, 
                           workerhosts=','.join(workerhosts),
                           jobname='worker',
                           index='$PBS_ARRAYID'
                           ) for i in range(len(workerhosts))], theano=False,
                           zero_index=True)
    ps_pbs = populate_template(name=experiment, joblist=[ps_jobstring.format(model=experiment, 
                           workerhosts=','.join(workerhosts),
                           jobname='ps',
                           index='0'
                           )], theano=False,
                           zero_index=True, gpu=False)
    
    write_pbs_file(experiment + '_worker.pbs', worker_pbs)
    write_pbs_file(experiment + '_ps.pbs', ps_pbs)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
