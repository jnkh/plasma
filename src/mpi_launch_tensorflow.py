from __future__ import print_function
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
task_num = comm.Get_size()

from hostlist import expand_hostlist
import tensorflow as tf
import socket,os

def get_host_to_id_mapping():
  return {host:i for (i,host) in enumerate(expand_hostlist( os.environ['SLURM_NODELIST']))}

def get_my_host_id():
  return get_host_to_id_mapping()[socket.gethostname()]


def get_host_list(port):
  return [ '{}:{}'.format(host,port) for host in expand_hostlist( os.environ['SLURM_NODELIST']) ]

def get_worker_host_list(base_port,workers_per_host):
  hosts = expand_hostlist( os.environ['SLURM_NODELIST'])
  ports = [base_port + i for i in range(workers_per_host)]
  l = []
  for h in hosts:
    for p in ports:
      l.append('{}:{}'.format(h,p))
  return l

def get_worker_host(base_port,workers_per_host,task_id):
  return get_worker_host_list(base_port,workers_per_host)[task_id]

def get_ps_host_list(base_port,num_ps):
  assert(num_ps < 10000000)
  port = base_port
  l = []
  hosts = expand_hostlist( os.environ['SLURM_NODELIST'])
  while True:
    for host in hosts:
      if len(l) >= num_ps:
        return l
      l.append('{}:{}'.format(host,port))
    port += 1  

def get_ps_host(base_port,num_ps,num_workers,task_id):
  return get_ps_host_list(base_port,num_ps)[task_id - num_workers]

def get_mpi_cluster_server_jobname():      
  NUM_GPUS = 4
  num_ps = 1
  num_ps_per_host = 1
  assert(num_ps_per_host >= num_ps)
  num_workers_per_host = NUM_GPUS
  num_hosts = len(get_host_list(1))
  num_workers = num_workers_per_host*num_hosts 
  
  num_total = num_workers + num_ps 
  assert(task_num > num_total)
  if task_index >= num_total: 
      exit(0)
  print('{}, task_id: {}, host_id: {}'.format(socket.gethostname(),task_index,get_my_host_id()))
  
  tasks_per_node = task_num / num_hosts
  task_index = task_index % tasks_per_node 
  
  if task_index < NUM_GPUS:
      job_name = 'worker'
      num_per_host = num_workers_per_host
  else:
      job_name = 'ps'
      task_index = task_index - NUM_GPUS 
      num_per_host = num_ps_per_host
  
  if task_index == 0:
      print('{} superfluous processes'.format(task_num - num_total))
  
  
  print('task id: {}'.format(task_index))
  print('hostname: {}'.format(socket.gethostname()))
  
  worker_hosts = get_worker_host_list(2222,num_workers_per_host)
  ps_hosts = get_ps_host_list(2322,num_ps)
  if task_index == 0:
      print('ps_hosts: {}\n, worker hosts: {}\n'.format(ps_hosts,worker_hosts))
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=get_my_host_id()*num_per_host+task_index)
  
  return cluster,server,job_name
