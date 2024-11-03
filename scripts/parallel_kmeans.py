import numpy as np
from scipy.spatial.distance import euclidean
from mpi4py import MPI
from sklearn.datasets import make_blobs

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_processes = comm.Get_size()

N_CLUSTERS = 8
N_DIM = 2


# we set node 0 to be the "leader" here.
if my_rank == 0:
    # generate simulated data
    true_centers = [(np.random.uniform() for _ in range(N_DIM)) for i in range(N_CLUSTERS)]
    mtx, ys = make_blobs(n_samples=100, cluster_std = [0.1 for _ in range(N_DIM)], centers=true_centers, n_features=N_DIM)
    # split up the dataset
    split = np.array_split(mtx, n_processes)
else:
    mtx = None
    split = None

split_data = comm.scatter(split, root=0) # send each chunk of data to a process; the data a process gets corresponds to its rank.

if my_rank == 0:
    centroid_idxs = np.random.randint(mtx.shape[1], size=N_CLUSTERS)
    centroids = mtx[centroid_idxs, :]
else:
    centroids = None

terminate = comm.bcast(False, root=0)
curr_centroids = comm.bcast(centroids, root=0)
old_score = 10e8

while not(terminate):
    labels = np.array([np.argmin(euclidian(centroid, pt) for centroid in curr_centroids) for pt in split_data])
    score = np.mean([euclidian(centroids[c], pt) for c, pt in zip(labels, split_data)])
    n_scores = len(labels)    
    avg_of_cluster = np.zeros((N_CLUSTERS, N_DIM))
    n_in_cluster = np.zeros((N_CLUSTERS)
    for cluster in range(N_CLUSTERS):
        pts_in_cluster = [row for label, row in zip(labels, mtx[:, ] if label == cluster]
        n_in_cluster[cluster] = len(pts_in_cluster)
        avg_of_cluster[cluster] = np.mean(pts_in_cluster)
    
    all_avgs = comm.gather(avg_of_cluster, root=0)
    all_n = comm.gather(n_in_cluster, root=0)
    len_scores = comm.gather(n_scores, root=0)
    all_scores = comm.gather(score, root=0)  
    if my_rank == 0:
        new_score = np.sum(
            [len_scores[p_rank] * all_scores[p_rank] for p_rank in range(n_processes)]
        ) / np.sum(
            [len_scores[p_rank] for p_rank in range(n_processes)]
        )
        if old_score - new_score <= 0.0001:
            print('score' + str(old_score - new_score))
            terminate = True
        old_score = new_score
        new_centroids = np.zeros((N_CLUSTERS, N_DIM))
        for cluster in range(N_CLUSTERS):
            true_mean = np.sum(
                [all_avgs[p_rank][cluster] * all_n[p_rank][cluster] for p_rank in range(n_processes)]
            ) / np.sum(
                [all_n[p_rank][cluster] for p_rank in range(n_processes)]
            )
            new_centroids[cluster, :] = true_mean
    curr_centroids = comm.bcast(new_centroids, root=0)
    terminate = comm.bcast(terminate, root=0)
    print(terminate)
