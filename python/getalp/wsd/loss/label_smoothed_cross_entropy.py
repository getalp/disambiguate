from torch.nn import Module
from torch.nn.functional import log_softmax
from torch import Tensor


class LabelSmoothedCrossEntropyCriterion(Module):

    def __init__(self, label_smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.eps = label_smoothing
        self.ignore_index = ignore_index

    # output : N x vocab_size
    # target : N              [0, vocab_size[
    def forward(self, output: Tensor, target: Tensor):
        output = log_softmax(output, dim=1)
        non_pad_mask = target.ne(self.ignore_index)
        nll_loss = -output.gather(dim=1, index=target.unsqueeze(1))[non_pad_mask]
        smooth_loss = -output.sum(dim=1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
        eps_i = self.eps / output.size(1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss
