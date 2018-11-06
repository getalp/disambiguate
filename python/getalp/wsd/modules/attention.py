from torch.nn import Module, Parameter
from torch.nn.functional import softmax, tanh
from math import sqrt
import torch

if torch.cuda.is_available():
    FloatTensorType = torch.cuda.FloatTensor
    LongTensorType = torch.cuda.LongTensor
else:
    FloatTensorType = torch.FloatTensor
    LongTensorType = torch.LongTensor


class Attention(Module):

    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(FloatTensorType(1, in_features))
        stdv = 1. / sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        h = torch.transpose(inputs, 1, 2)
        u = tanh(h)
        u = torch.matmul(self.weight, u)
        #        u = u.view(u.size(1))
        a = softmax(u, dim=2)
        #        a = a.unsqueeze(2)
        a = torch.transpose(a, 1, 2)
        c = torch.matmul(h, a)
        c = c.transpose(1, 2)
        ones = torch.ones((inputs.size(0), inputs.size(1), 1)).type(FloatTensorType)
        c = torch.matmul(ones, c)

        return torch.cat((inputs, c), dim=2)
