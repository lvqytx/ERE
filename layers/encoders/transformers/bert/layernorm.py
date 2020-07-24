import  math
import torch
import torch.nn as nn
import torch.nn.functional as F
class ConditionalLayerNorm(nn.Module):

    def __init__(self,hidden_size, eps=1e-12):
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.variance_epsilon = eps

        self.beta_dense = Linear(hidden_size*2, hidden_size, bias=False)
        self.gamma_dense = Linear(hidden_size*2, hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        bias = self.bias + beta
        weight = self.weight + gamma

        u = x.mean(-1, keepdim=True)
        s = (x-u).pow(2).mean(-1, keepdim=True)
        x = (x-u) / torch.sqrt(s+self.variance_epsilon)
        return weight * x + bias

# nn Linear实现
class Linear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a= math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            nn.init.uniform_(self.bias,-bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )