# encoding: utf-8
import torch
import math
from torch.nn import Module
from . import resnet_care as resnet
from .trans import Attention
from .trans import Cross_Attention
from .trans import TransStack
import torch.nn as nn
import ipdb


class CARE(Module):
    def __init__(self, param_momentum, total_iters):
        super(CARE, self).__init__()
        self.total_iters = total_iters
        self.param_momentum = param_momentum
        self.current_train_iter = 0

        self.student_encoder = resnet.resnet50(
            low_dim=256, width=1, hidden_dim=4096, MLP="care", CLS=False, bn="torchsync", predictor=True
        )
        self.teacher_encoder = resnet.resnet50(
            low_dim=256, width=1, hidden_dim=4096, MLP="care", CLS=False, bn="torchsync", predictor=False
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.lambda_fea = 100
        self.fea_loss = torch.nn.MSELoss()
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.momentum_update(m=0)

    @torch.no_grad()
    def momentum_update(self, m):
        for p1, p2 in zip(self.student_encoder.named_parameters(), self.teacher_encoder.named_parameters()):
            flag = 'fc' in p1[0]
            if not flag:
                # p2.data.mul_(m).add_(1 - m, p1.detach().data)
                p2[1].data = m * p2[1].data + (1.0 - m) * p1[1].detach().data

        for p1, p2 in zip(self.student_encoder.fc1.parameters(),
                          self.teacher_encoder.fc1.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data
        for p1, p2 in zip(self.student_encoder.fc2.parameters(),
                          self.teacher_encoder.fc2.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

    def get_param_momentum(self):
        return 1.0 - (1.0 - self.param_momentum) * (
            (math.cos(math.pi * self.current_train_iter / self.total_iters) + 1) * 0.5
        )

    def forward(self, inps, update_param=True):
        if update_param:
            current_param_momentum = self.get_param_momentum()
            self.momentum_update(current_param_momentum)

        x1, x2 = inps[0], inps[1]
        q1, att_q1, fea_q1 = self.student_encoder(x1)
        q2, att_q2, fea_q2 = self.student_encoder(x2)

        with torch.no_grad():
            k1, att_k1, fea_k1 = self.teacher_encoder(x2)
            k2, att_k2, fea_k2 = self.teacher_encoder(x1)

        con_loss = (4 - 2 * ((q1 * k1).sum(dim=-1, keepdim=True) + (q2 * k2).sum(dim=-1, keepdim=True))).mean()
        con2_loss = (4 - 2 * ((att_q1 * att_k1).sum(dim=-1, keepdim=True) + (att_q2 * att_k2).sum(dim=-1, keepdim=True))).mean()

        fea_loss = self.fea_loss(q1, att_q1.detach()) + self.fea_loss(q2, att_q2.detach()) + self.fea_loss(k1, att_k1.detach()) + self.fea_loss(k2, att_k2.detach())

        loss = con_loss + con2_loss + self.lambda_fea * fea_loss
        self.current_train_iter += 1
        if self.training:
            return loss, con_loss, con2_loss, self.lambda_fea * fea_loss
