import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class GaussianNoise(object):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, pic):
        arra = np.array(pic)
        noises = np.random.normal(0, self.strength, arra.shape)
        noises = np.uint8(noises)
        arra += noises
        pic = Image.fromarray(arra)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

# https://arxiv.org/pdf/1904.05068 Not sure about this?
class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def get_auxiliary_data(datas, transformations):
    data_list = [[] for _ in range(len(transformations))]
    for data in datas:
        for i, trans in enumerate(transformations):
            trans_sample = trans(data.cuda())
            data_list[i].append(torch.unsqueeze(trans_sample, dim=0))
    for i in range(len(transformations)):
        data_list[i] = torch.cat(data_list[i], dim=0).cuda()
    return data_list