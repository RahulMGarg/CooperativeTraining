from jinja2 import Template
import os
import argparse
from itertools import product

CUDA_DIRECTORY = "/global/scratch/jasonhar2/cuda/"
CUDA_ASSIGNER = "/global/scratch/jasonhar2/cuda-assigner/"

ada_template = """#!/bin/sh
# Script for running serial program, diffuse.
#PBS -S /bin/bash
#PBS -q {% if gpu %}nvidia{% else %}batch{% endif %}
#PBS -l walltime={{ walltime }}
#PBS -l mem={{ memory }}
#PBS -j eo
#PBS -N {{ name }}
#PBS -t 1-{{ num_jobs }}

cd {{ path }}
{% if gpu %}export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"{{ cuda_path }}lib64"
export C_INCLUDE_PATH=$C_INCLUDE_PATH:"{{ cuda_path }}include"
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:"{{ cuda_path }}include"
export PATH=$PATH:"{{ cuda_path }}bin"{% endif %}

eval $(python {{ cuda_assigner }}run_client.py --request --jobid $PBS_ARRAYID)
{% if theano %}export THEANO_FLAGS="device=gpu0,floatX=float32,base_compiledir=/global/scratch/jasonhar/.theano/{{ name }}-$PBS_ARRAYID"{% endif %}
{% for item in preamble %}{{ item }}{% endfor %}
case $PBS_ARRAYID in
{% for job in joblist %}{{ job[0] }}) {{ job[1] }};;
{% endfor %}*) echo "Unrecognized jobid '$PBS_ARRAYID'"
esac

{% if theano %}
\\rm -r /global/scratch/jasonhar/.theano/simple_exp_nnet_1000-$PBS_ARRAYID
{% endif %}eval $(python {{ cuda_assigner }}run_client.py --release --jobid $PBS_ARRAYID)
"""

def format_joblist(joblist):
    return [(i+1,j) for i,j in enumerate(joblist)]

def populate_template(name, joblist, path=os.getcwd(), gpu=True, 
                      walltime='36:00:00', memory='2000mb', theano=True,
                      preamble=[], cuda_path=CUDA_DIRECTORY, cuda_assigner=CUDA_ASSIGNER):
    args = locals()
    args['num_jobs'] = len(joblist)
    args['joblist'] = format_joblist(joblist)
    template = Template(ada_template)
    return template.render(args)

def combine_lists(list_of_lists, output_list = []):
    if len(list_of_lists) == 0:
        return output_list
    else:
        current_list = list_of_lists.pop()
        updated_list = []
        for c in current_list:
            for o in output_list:
                updated_list.append(o.append())
        print(current_list)
        return combine_lists(list_of_lists, output_list)

def parse_joblist(joblist):
    split_list = joblist.split('{{')
    lists = [i.split('}}')[0] for i  in split_list[1:]]
    commands = [split_list[0]] + [i.split('}}')[1] for i  in joblist.split('{{')[1:]]
    commands = '%s'.join(commands)
    out = []
    for i in lists:
        if len(i.split(':')) > 1:
            out.append(range(*[int(j) for j in i.split(':')]))
        else:
            out.append(eval('[' + i + ']'))
    return [commands % i for i in product(*out)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=str,
        default="",
        help="Job string template. Of the form: 'python foo.py --seed {{0:5}} --models \
              {{\"a\", \"b\", \"c\"}}' which will generate jobs from the cartesian \
              product of [0, 1, ...,  4] and ['a', 'b', 'c']."
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the experiment (required). The jobfile will be saved to NAME.pbs in the current directory."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the PBS file to stdout instead of saving it to a PBS file"
    )

    parser.add_argument(
        "-f",
        action="store_false",
        help="Force overwriting of existing PBS files"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run job on the GPU queue. Adds CUDA imports to your path."
    )

    parser.add_argument(
        "--theano",
        action="store_true",
        help="Job requires theano"
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

    args, unparsed = parser.parse_known_args()
    joblist = parse_joblist(args.jobs)

    pbs_text = populate_template(args.name, joblist=joblist, gpu=args.gpu, 
                            theano=args.theano, memory=str(args.memory) +'mb',
                            walltime=args.walltime)
    if args.check:
        print(pbs_text)
    else:
        filename = args.name + '.pbs'
        # Confirm overwiting. If the -f flag is used, this check will be ignored.
        if os.path.exists(filename) and args.f:
            response = raw_input("PBS file %s already exists. Overwrite? Y / N \n" % filename)
            if response.lower() == 'n':
                exit()
        with open(filename, 'w') as f:
            f.write(pbs_text)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))