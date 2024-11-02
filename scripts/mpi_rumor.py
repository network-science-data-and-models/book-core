import networkx as nx
import numpy as np
import pickle
import sys
print(sys.executable)
from mpi4py import MPI

def rumor_model_agent_based(G, alpha=0.2, beta=0.1, n_seeds=5, n_iter=500):
    """
    Does an agent-based rumor model simulation on a given network.
    
    Inputs:
    --------------
    G: networkx graph object
    alpha: parameter that governs conversion from ignorant to spreader; float in range (0, 1.0]
    beta: parameter that governs conversion from spreader to stifler; float in range (0, 1.0]
    n_seeds: number of initial infected nodes; int.
    n_iter: number of iterations to run the model for; int.

    Returns: 
    -------------
    States of each node at each time step
    """    
    state_dict = {n: 'ignorant' for n in G.nodes}
    seed_nodes = np.random.choice(list(G.nodes), n_seeds, replace=False)
    for seed in seed_nodes:
        state_dict[seed] = 'spreader'
    
    state_dict_list = []
    for i in range(n_iter):
        state_dict_list.append(state_dict)
        state_dict_new = state_dict.copy()
        spreaders = [n for n, state in state_dict.items() if state == 'spreader']
        stiflers = [n for n, state in state_dict.items() if state == 'stifler']
        for spreader in spreaders:
            for neighbor in G.neighbors(spreader):
                if state_dict[neighbor] == 'ignorant' and np.random.uniform() <= alpha:
                    state_dict_new[neighbor] = 'spreader'
                elif state_dict[neighbor] == 'spreader' and np.random.uniform() <= beta:
                    if np.random.uniform() > 0.5:
                        state_dict_new[neighbor] = 'stifler'
                    else:
                        state_dict_new[spreader] = 'stifler'
                elif state_dict[neighbor] == 'stifler' and np.random.uniform() <= beta:
                    state_dict_new[spreader] = 'stifler'
        state_dict = state_dict_new
    
    return state_dict_list

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
G = nx.watts_strogatz_graph(250000, 6, 0.05)

pickle.dump(rumor_model_agent_based(G), open('rumor_model_agent_based_process_{}.pkl'.format(str(rank)), 'wb'))
