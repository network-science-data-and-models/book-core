from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'whales': True}
else:
    data = None

print('I am process ', rank, 'and my data is ', data)

data = comm.bcast(data, root=0)
print('I am process ', rank, 'and my data is now ', data)

ranks = comm.gather(rank, root=0)
if rank == 0:
    print('I am process ', rank, 'and the ranks of all processes are ', ranks)
