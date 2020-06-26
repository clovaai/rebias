"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

ReBias model wrapper.
"""
import torch.nn as nn


class ReBiasModels(object):
    """A container for the target network and the intentionally biased network.
    """
    def __init__(self, f_net, g_nets):
        self.f_net = f_net
        self.g_nets = g_nets

    def to(self, device):
        self.f_net.to(device)
        for g_net in self.g_nets:
            g_net.to(device)

    def to_parallel(self, device):
        self.f_net = nn.DataParallel(self.f_net.to(device))
        for i, g_net in enumerate(self.g_nets):
            self.g_nets[i] = nn.DataParallel(g_net.to(device))

    def load_models(self, state_dict):
        self.f_net.load_state_dict(state_dict['f_net'])
        for g_net, _state_dict in zip(self.g_nets, state_dict['g_nets']):
            g_net.load_state_dict(_state_dict)

    def train_f(self):
        self.f_net.train()

    def eval_f(self):
        self.f_net.eval()

    def train_g(self):
        for g_net in self.g_nets:
            g_net.train()

    def eval_g(self):
        for g_net in self.g_nets:
            g_net.eval()

    def train(self):
        self.train_f()
        self.train_g()

    def eval(self):
        self.eval_f()
        self.eval_g()

    def forward(self, x):
        f_pred, f_feat = self.f_net(x)
        g_preds, g_feats = [], []
        for g_net in self.g_nets:
            _g_pred, _g_feat = g_net(x)
            g_preds.append(_g_pred)
            g_feats.append(_g_feat)

        return f_pred, g_preds, f_feat, g_feats

    def __call__(self, x):
        return self.forward(x)
