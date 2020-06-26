"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

De-biasing comparison methods.
Cadene, Remi, et al. "RUBi: Reducing Unimodal Biases for Visual Question Answering.",
Clark, Christopher, Mark Yatskar, and Luke Zettlemoyer. "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases.", EMNLP 2019.

Reference codes:
    - https://github.com/cdancette/rubi.bootstrap.pytorch/blob/master/rubi/models/criterions/rubi_criterion.py
    - https://github.com/chrisc36/debias/blob/master/debias/modules/clf_debias_loss_functions.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradMulConst(torch.autograd.Function):
    """ This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)


class RUBi(nn.Module):
    """RUBi
    Cadene, Remi, et al. "RUBi: Reducing Unimodal Biases for Visual Question Answering.",
    Advances in Neural Information Processing Systems. 2019.
    """
    def __init__(self, question_loss_weight=1.0, **kwargs):
        super(RUBi, self).__init__()
        self.question_loss_weight = question_loss_weight
        self.fc = nn.Linear(kwargs.get('feat_dim', 128), kwargs.get('num_classes', 10)).cuda()

    def forward(self, f_feat, g_feat, labels, f_pred, **kwargs):
        """Compute RUBi loss.

        Parameters
        ----------
        f_feat: NOT USED (for compatibility with other losses).
        g_feat: features from biased network (will be passed to `self.fc` for computing `g_pred`)
        labels: class labels
        f_pred: logit values from the target network
        """
        g_feat = g_feat.view(g_feat.shape[0], -1)
        g_feat = grad_mul_const(g_feat, 0.0)  # don't backpropagate through bias encoder
        g_pred = self.fc(g_feat)
        logits_rubi = f_pred * torch.sigmoid(g_pred)
        fusion_loss = F.cross_entropy(logits_rubi, labels)
        question_loss = F.cross_entropy(g_pred, labels)
        loss = fusion_loss + self.question_loss_weight * question_loss

        return loss


class LearnedMixin(nn.Module):
    """LearnedMixin + H
    Clark, Christopher, Mark Yatskar, and Luke Zettlemoyer.
    "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases.",
    EMNLP 2019.
    """
    def __init__(self, w=0.36, **kwargs):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        self.fc = nn.Linear(kwargs.get('feat_dim', 128), 1).cuda()

    def forward(self, f_feat, g_pred, labels, f_pred):
        f_feat = f_feat.view(f_feat.shape[0], -1)
        f_pred = f_pred.view(f_pred.shape[0], -1)
        g_pred = g_pred.view(g_pred.shape[0], -1)

        factor = self.fc.forward(f_feat)
        factor = F.softplus(factor)
        g_pred *= factor

        loss = F.cross_entropy(f_pred+g_pred, labels)

        bias_lp = F.log_softmax(g_pred, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean()

        return loss + self.w * entropy
