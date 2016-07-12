from __future__ import print_function
from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
import time,sys
import mpipool as mp
from pylab import *
#pool._processes
print ("hello world from process {}".format( my_rank))

def test(x):
    time.sleep(1)
    return x


def test_mm(x):
    a = randn(5000,5000)
    return sum(a**5)

test_fn = test_mm
num_tests = 64


pool = mp.MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0) 
   

print('parallel')
start = time.time()
for res in pool.imap(test_fn,range(num_tests)):
    print(res)

print('parallel elapsed: {}'.format(time.time() - start))
    #print(res)

print('serial')
start = time.time()
res = map(test_fn,range(num_tests))
print('elapsed: {}'.format(time.time() - start))
#print(res)

pool.close()
sys.exit()


