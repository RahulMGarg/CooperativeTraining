# CooperativeTraining

Train model asynchronously on multiple GPUs in a cooperative fashion using tensor flow.

To run this code you'll need access to a cluster running Torque and a machine to run [redis](https://redis.io).

Instructions for running jobs.
1. Set the hostname of the machine running your redis server in in torque_manager.py and start your redis server.
2. Use the python script distributed_tf_manager.py to generate your PBS job file. You can then launch this job manually or by setting up the ssh settings appropriately, you can have the script launch it automatically using the --launch flag.

All management of the redis queue and launching of the tensorflow workers is handled by torque_manager.py which can be run manually for testing purposes. 

Individual distributed optimization algorithms are contained in hogwild.py and cooptimization.py.
