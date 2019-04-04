# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:40:07 2015

Copyright 2015 Nicolo' Navarin

This file is part of scikit-learn-graph.

scikit-learn-graph is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

scikit-learn-graph is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""
from .WLGraphNodeKernel import WLGraphNodeKernel
from sklearn.preprocessing import normalize
class WLVectorizer():
    """
    Transforms labeled, weighted, nested graphs in sparse 
    vectors using ODDST graph kernel representation.
    """
    def __init__(self, r = 3, normalization = True, hash=None):
                         self.vectObject=WLGraphNodeKernel(
                                           r ,
                                           normalization, hash=hash)
                         self.normalization=normalization

    def transform(self, G_list):
        """
        Returns: a list lists of length r of lists of length r of matrices of size n_vertices x n_features
        """
        #Transform returns a sparse matrix of feature vectors not normalized
        return self.vectObject.transform(G_list)
#         if self.normalization:
#             ve = normalize(ve, norm='l2', axis=1)
#         return ve
  #  def transform_incr(self, G_list):
        #Transform returns a sparse matrix of feature vectors not normalized
  #       return self.vectObject.transform_incr(G_list)
    def getnfeatures(self):
        return self.vectObject.getnfeatures()
#         if self.normalization:
#             ve = normalize(ve, norm='l2', axis=1)
#         return ve
            
"""
        Parameters
        ----------
        r : int 
            The maximal radius size.

        l : float
            The lambda weight factor.

        nbits : int 
            The number of bits that defines the feature space size: |feature space|=2^nbits.

        normalization : bool 
            If set the resulting feature vector will have unit euclidean norm.

    
            If 0 then treat all labels as strings. 

        discretization_dimension : int
            Size of the discretized label vector.
"""
