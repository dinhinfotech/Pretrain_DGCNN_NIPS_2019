__author__ = "Riccardo Tesselli"
__date__ = "04/feb/2015"
__credits__ = ["Riccardo Tesselli"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Riccardo Tesselli"
__email__ = "riccardo.tesselli@gmail.com"
__status__ = "Production"

import numpy as np
from operator import itemgetter
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import dok_matrix

#def _dict_to_csr(term_dict):
#    term_dict_v = list(term_dict.itervalues())
#    term_dict_k = list(term_dict.iterkeys())
#    print term_dict_k
#    shape = list(np.repeat(np.asarray(term_dict_k).max() + 1,2))
#    csr = csr_matrix((term_dict_v, zip(*term_dict_k)), shape = shape)
#    return csr
#    
#def myconvert(term_dict):
#    ''' Convert a dictionary with elements of form ('d1', 't1'): 12 to a CSR type matrix.
#    The element ('d1', 't1'): 12 becomes entry (0, 0) = 12.
#    * Conversion from 1-indexed to 0-indexed.
#    * d is row
#    * t is column.
#    '''
#    # Create the appropriate format for the COO format.
#    data = []
#    row = []
#    col = []
#    for k, v in term_dict.items():
#        r = int(k[0][1:])
#        c = int(k[1][1:])
#        data.append(v)
#        row.append(r-1)
#        col.append(c-1)
#    # Create the COO-matrix
#    coo = coo_matrix((data,(row,col)))
#    # Let Scipy convert COO to CSR format and return
#    return csr_matrix(coo)

def computeFeaturesWeights(svindexes,coeflist,dictfeatures):
    """
    Function that computes the relevance score w_j=sum_over_support_graphs_of(alpha*y*phi(G)_j) 
    """
    features=[c for (r,c) in dictfeatures.keys()]
    features=np.unique(features)
    weights={}
    for f in features:
        weight=0
        coefindex=0
        for svi in svindexes:
            if not dictfeatures.get((svi,f)) is None:
                weight+=coeflist[coefindex]*dictfeatures.get((svi,f))
            coefindex+=1
        weights[f]=weight
    return weights

def topWeights(number,weights,positive=True):
    listweights=weights.items()
    listweights.sort(key=itemgetter(1))
    if positive:
        listweights=listweights[-number:]
    else:
        listweights=listweights[:number]
    return dict(listweights)
    
def convert_to_sparse_matrix_enc(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        if not MapEncToId is None:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( MapEncToId[j] )
            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
            return X, MapEncToId
        else:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( j )
            MapEncToId={}
            idenc=0
            for enc in np.unique(col):
                MapEncToId[enc]=idenc
                idenc+=1
            colid=[]
            for enc in col:
                colid.append(MapEncToId[enc])
            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1))
            return X, MapEncToId


def new_convert_to_sparse_matrix(feature_dict,N,hash):
    #hash 100
    M=max(feature_dict.keys())+1
    Mat=dok_matrix((M, hash))
    #print(M, N)
    for i in feature_dict.keys():
        for j in feature_dict[i].keys():
            Mat[i, j%hash] += feature_dict[i][j]
    return Mat.tocsr()

def _new_convert_to_sparse_matrix(feature_dict,N):
    #hash 100
    M=max(feature_dict.keys())+1
    Mat=dok_matrix((M, N))
    #print(M, N)
    for i in feature_dict.keys():
        for j in feature_dict[i].keys():
            Mat[i, j] = feature_dict[i][j]
    return Mat.tocsr()

def convert_to_sparse_matrix(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector

        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values

        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        if not MapEncToId is None:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( MapEncToId[j])
            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
            return X
        else:
            for i, j in feature_dict.iterkeys():
                row.append( i )
                col.append( j )
            MapEncToId={}
            idenc=0
            for enc in np.unique(col):
                MapEncToId[enc]=idenc
                idenc+=1
            colid=[]
            for enc in col:
                colid.append(MapEncToId[enc])
            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1))
            return X
            
def convert_to_sparse_matrix_for_deep(feature_dict, MapEncToId=None):
        """
        Function that convert the feature vector from dictionary to sparse matrix
        @type feature_dict: Dictionary
        @param feature_dict: a feature vector
        
        @type MapEncToId: self.UniqueMap
        @param MapEncToId: Map between feature's encodings and integer values
        
        @rtype: scipy.sparse.csr_matrix
        @return: the feature vector in sparse form
        """
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
#        if not MapEncToId is None:
#            for i, j in feature_dict.iterkeys():
#                row.append( i )
#                col.append( MapEncToId[j]) 
#            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1))
#            return X
#        else:
        for i, j in feature_dict.iterkeys():
            row.append( i )
            col.append( j )
        #MapEncToId={}
        idenc=0
        for enc in np.unique(col):
            MapEncToId[enc]=idenc
            idenc+=1
        colid=[]
        for enc in col:
            colid.append(MapEncToId[enc])
        X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1))
        return X
#def convert_to_sparse_matrix_list(feature_dict, MapEncToId=None):
#        """
#        Function that convert the feature vector from dictionary to sparse matrix
#        @type feature_dict: Dictionary
#        @param feature_dict: a feature vector
#        
#        @type MapEncToId: self.UniqueMap
#        @param MapEncToId: Map between feature's encodings and integer values
#        
#        @rtype: scipy.sparse.csr_matrix
#        @return: the feature vector in sparse form
#        """
#        if len(feature_dict) == 0:
#            raise Exception('ERROR: something went wrong, empty feature_dict.')
#        data = feature_dict.values()
#        row, col = [], []
#        if not MapEncToId is None:
#            for i, j in feature_dict.iterkeys():
#                row.append( i )
#                col.append( MapEncToId[j]) 
#            X = csr_matrix( (data,(row,col)), shape = (max(row)+1, max(col)+1), dtype=np.dtype(object))
#            return X
#        else:
#            for i, j in feature_dict.iterkeys():
#                row.append( i )
#                col.append( j )
#            MapEncToId={}
#            idenc=0
#            for enc in np.unique(col):
#                MapEncToId[enc]=idenc
#                idenc+=1
#            colid=[]
#            for enc in col:
#                colid.append(MapEncToId[enc])
#            X = csr_matrix( (data,(row,colid)), shape = (max(row)+1, max(colid)+1), dtype=np.dtype(object))
#            return X