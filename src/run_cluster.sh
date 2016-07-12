python cluster.py --job_name=ps --ps_hosts=localhost:2222 --worker_hosts=localhost:2224,localhost:2225,localhost:2226 --task_index=0 &
python cluster.py --job_name=worker  --ps_hosts=localhost:2222 --worker_hosts=localhost:2224,localhost:2225,localhost:2226 --task_index=0 &
python cluster.py --job_name=worker  --ps_hosts=localhost:2222 --worker_hosts=localhost:2224,localhost:2225,localhost:2226 --task_index=1  &
python cluster.py --job_name=worker  --ps_hosts=localhost:2222 --worker_hosts=localhost:2224,localhost:2225,localhost:2226 --task_index=2  &
#CUDA_VISIBLE_DEVICES='' 
