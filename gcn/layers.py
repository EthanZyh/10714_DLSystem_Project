from needle.nn import Parameter, Module
from needle.autograd import Tensor
from needle import ops
from needle import init
import math

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.rand(
                self.in_features,
                self.out_features,
                low=-1.0 / math.sqrt(out_features),
                high=1.0 / math.sqrt(out_features),
                device=device,
                dtype=dtype,
                requires_grad=True,
            ))
        if bias:
            self.bias = Parameter(init.rand(
                1,
                self.out_features,
                low=-1.0 / math.sqrt(out_features),
                high=1.0 / math.sqrt(out_features),
                device=device,
                dtype=dtype,
                requires_grad=True,
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor, adj: Tensor) -> Tensor:
        support = ops.matmul(input, self.weight)
        output = ops.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias.broadcast_to(output.shape)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'