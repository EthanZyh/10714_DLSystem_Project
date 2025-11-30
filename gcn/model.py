from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from needle import nn
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = ops.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x
    
