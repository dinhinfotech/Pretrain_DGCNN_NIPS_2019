import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import pairwise_kernels
from GraphKernels.WLVectorizer import WLVectorizer
from scipy.sparse import csr_matrix, vstack, hstack
from numpy import stack
from sklearn.preprocessing import normalize


def computeWL(graph_list_dortmund, h=3, hash=100, separated_iterations=False):
    g_list=[]
    for i in graph_list_dortmund:
        #print(i.x)
        g = nx.Graph()
        node_labels=np.argmax(i.x, axis=1)
        for n in range(node_labels.shape[0]):
            #print("node",n)
            g.add_node(n, label=node_labels[n])

        g.graph['node_order'] = range(node_labels.shape[0])
        #print(i.edge_index[0], i.edge_index[1])
        for e in range(i.edge_index.shape[1]):
            first=i.edge_index[0,e].item()
            second=i.edge_index[1,e].item()
            g.add_edge(first, second)
        g_list.append(g)

    if separated_iterations:
        features= vectorize_wl_separated(g_list, h, hash)
    else:
        features= vectorize_wl(g_list, h, hash)


    #n=norm(features, axis=None)
    #print(n)
    #print("----")
    return features #, n #normalize,



def vectorize_wl(graphs=None, n_iter=None, hash=None):
    #features divided in the hash
    new_graphs = graphs #convert_graphs(graphs)

    WLvect= WLVectorizer(r=n_iter, hash=hash)
    iters_features = WLvect.transform(new_graphs)
    #print(iters_features)
    list_graph_vector = []
    for idx in range(len(graphs)):
        M = iters_features[0][idx]
        for iter_id in range(1, n_iter+1):
            M+=iters_features[iter_id][idx]
        #print(M)
        list_graph_vector.append(csr_matrix(M.sum(axis=0)))

    M = vstack(list_graph_vector)

    return M

def vectorize_wl_separated(graphs=None, n_iter=None, hash=None):
    #features divided in the hash
    new_graphs = graphs #convert_graphs(graphs)

    WLvect= WLVectorizer(r=n_iter, hash=hash//(n_iter+1))
    iters_features = WLvect.transform(new_graphs)
    #print(iters_features)
    list_graph_vector = []
    for idx in range(len(graphs)):
        #M = iters_features[0][idx]
        #for iter_id in range(1, n_iter+1):
            #M.stack(iters_features[iter_id][idx])
        M=hstack([iters_features[iter_id][idx] for iter_id in range(0, n_iter+1)])
        #print(M)
        list_graph_vector.append(csr_matrix(M.sum(axis=0)))

    M = vstack(list_graph_vector)

    return M


def wl(graphs=None, n_iter=None):
    new_graphs = convert_graphs(graphs)

    print("Done converting graphs")

    WLvect= WLVectorizer(r=n_iter)
    iters_features = WLvect.transform(new_graphs)

    list_graph_vector = []
    for idx in range(len(graphs)):
        M = iters_features[0][idx]
        for iter_id in range(1, n_iter+1):
            M+= iters_features[iter_id][idx]
        list_graph_vector.append(csr_matrix(M.sum(axis=0)))
    M = vstack(list_graph_vector)

    """--"""
    G = pairwise_kernels(M)
    print("Done computing kernel")
    N = G.shape[0]
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])

    for idx in range(N):
        G[idx, idx] = 1.0

    return G


