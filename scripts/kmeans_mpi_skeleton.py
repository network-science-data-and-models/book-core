import numpy as np
from scipy.spatial.distance import euclidean
from mpi4py import MPI
from sklearn.datasets import make_blobs

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
print('Hello from process ', my_rank)
n_processes = comm.Get_size()

N_CLUSTERS = 5
N_DIM = 2


# we set node 0 to be the "leader" here.
if my_rank == 0:
    # generate simulated data
    true_centers = [[np.random.uniform(-50, 50) for _ in range(N_DIM)] for i in range(N_CLUSTERS)]
    true_centers = np.array(true_centers)
    print('true centers:')
    print(true_centers)
    mtx, ys = make_blobs(n_samples=1000, cluster_std = [0.05 for _ in range(N_CLUSTERS)], centers=true_centers, n_features=N_DIM)
    # split up the dataset
    split = np.array_split(mtx, n_processes)
else:
    mtx = None
    split = None

# we're going to do 10 runs (in case of crappy centroid initalization) and pick the best one.
best_score = 10e8
best_centroids = None

for _ in range(10):
    split_data = comm.scatter(split, root=0) # send each chunk of data to a process; the data a process gets corresponds to its rank.
    
    # the leader process picks N_CLUSTERS random points as starting centroids.
    if my_rank == 0:
        centroid_idxs = np.random.randint(len(mtx), size=N_CLUSTERS)
        centroids = mtx[centroid_idxs, :]
    else:
        centroids = None
    
    # we terminate when we stop improving our total distance from the centroid.
    terminate = comm.bcast(False, root=0)
    # tell all processes about the starting centroids.
    curr_centroids = comm.bcast(centroids, root=0)
    # set an absurdly high number as our score to improve upon.
    old_score = 10e8
    
    while not(terminate):
        # label points with their closest centroid's index
        
        #############################
        # TODO: write code to assign each point a label!
        #############################

        # get the total distance for this process' set of points it's working on.
        score = np.sum([euclidean(curr_centroids[c, :], pt) for c, pt in zip(labels, split_data)])
        # keep track of the furthest point from the centroid (that this process knows about).
        # we use this to re-initialize any clusters that turn up empty.
        furthest_point = sorted(
            [(pt, euclidean(curr_centroids[c, :], pt)) for c, pt in zip(labels, split_data)],
            key=lambda b: b[1]
        )[-1]
        
        # make the cluster centroids
        avg_of_cluster = np.zeros((N_CLUSTERS, N_DIM))
        n_in_cluster = np.zeros(N_CLUSTERS)
        for cluster in range(N_CLUSTERS):
            pass
            #############################
            # TODO: determine the centroids (according to this process)
            # as well as the number of points that belong to each centroid.
            # get all points that belong in this cluster (based on our labeling)
            # count how many there are in each cluster
            # take the average of the points assigned to each centroid
            # if there are none, pick a convenient placeholder.
            #############################
    
        # gather up our cluster centroids from each process, along with the point counts to each cluster and the total distances.
        # also gather the furthest point each process knows about.
        all_avgs = comm.gather(avg_of_cluster, root=0)
        all_n = comm.gather(n_in_cluster, root=0)
        all_scores = comm.gather(score, root=0)  
        furthest_points = comm.gather(furthest_point, root=0)
        
        # the leader process will now compile the score & compute the new centroids
        if my_rank == 0:
            new_score = np.sum(all_scores) # total euclidean distance for all points from their assigned centroids
            print('score difference: ' + str(old_score - new_score))
            print('new score: ', new_score)
            print('old score: ', old_score)

            if old_score - new_score <= 0.0001: # if our score stops decreasing, we terminate before the next iteration.
                terminate = True

            old_score = new_score # update our best score

            # build new centroids
            new_centroids = np.zeros((N_CLUSTERS, N_DIM))
            for cluster in range(N_CLUSTERS):
                ##########################
                # TODO: count how many points were in the cluster (from each process)
                tot_points = 0
                ##########################
                
                # if there are no points, we assign the point that's furthest from any centroid to be a new centroid.
                if tot_points == 0:
                    new_centroids[cluster, :] = sorted(furthest_points, key=lambda b: b[1])[-1][0]
                else:
                    #########################
                    # TODO: otherwise, we take the weighted average of all centroids computed by the worker processes.
                    #########################
                    pass
                    true_mean = np.zeros(N_DIM)
                    new_centroids[cluster, :] = true_mean
        else:
            new_centroids = None

        # broadcast our new centroids to everyone and let them know if the while loop terminates.
        curr_centroids = comm.bcast(new_centroids, root=0)
        terminate = comm.bcast(terminate, root=0)
        print()
 
    # the leader process checks if our result is better than we've seen in other runs.
    if my_rank == 0:
        print(curr_centroids)
        if new_score < best_score:
            best_score = new_score
            best_centroids = curr_centroids

print(best_centroids, best_score)
