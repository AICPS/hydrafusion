import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor



'''Represents the base class Gating Module used for selecting branches of the hydranet for execution'''
class GateModule(nn.Module):
    def __init__(self):
        super(GateModule, self).__init__()



'''gate module that uses CNN layers followed by linear layers to identify the best branch'''
class DeepGatingModule(GateModule):
    def __init__(self, input_channels, output_shape, dropout):
        super(DeepGatingModule, self).__init__()
        self.c1 = nn.Conv2d(input_channels, 32, 5, 3, 1)
        self.bn1 = nn.BatchNorm2d(32, 32)
        self.c2 = nn.Conv2d(32, 16, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(16, 16)
        self.c3 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(8, 8)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.l1 = nn.Linear(3584, output_shape)
        self.activation = F.relu
        self.dropout = dropout


    def forward(self, x):
        x = self.activation(self.bn1(self.c1(x)))
        x = self.activation(self.bn2(self.c2(x)))
        x = self.activation(self.bn3(self.c3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.maxpool(x)
        x = torch.flatten(x)
        return self.l1(x)



'''attention based gating module that uses a self-attention layer as part of its gating mechanism.'''
class AttentionGatingModule(GateModule):
    def __init__(self, input_channels, output_shape, dropout):
        super(AttentionGatingModule, self).__init__()
        self.c1 = nn.Conv2d(input_channels, 32, 5, 3, 1)
        self.bn1 = nn.BatchNorm2d(32, 32)
        self.c2 = nn.Conv2d(32, 16, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(16, 16)
        self.self_attn = SelfAttention(8)
        self.c3 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(8, 8)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.l1 = nn.Linear(3584, output_shape)
        self.activation = F.relu
        self.dropout = dropout


    def forward(self, x):
        x = self.activation(self.bn1(self.c1(x)))
        x = self.activation(self.bn2(self.c2(x)))
        x = self.activation(self.bn3(self.c3(x)))
        x = self.activation(self.self_attn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.maxpool(x)
        x = torch.flatten(x)
        return self.l1(x)



"Self attention layer for `n_channels`. from: https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00"
class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))


    def _conv(self,n_in,n_out):
        return nn.Conv1d(n_in, n_out, kernel_size=1,  bias=False)


    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()



'''This module implements static gating. Depending on where the scene was filmed (city, fog, rain, snow, etc.) the gate selects a fixed set of branches matching that scene. '''
class KnowledgeBasedGateModule(GateModule):
    def __init__(self):
        super(KnowledgeBasedGateModule, self).__init__()
        self.branch_names = ['radar', 'camera_left', 'camera_right', 'lidar', 'camera_both', 'camera_lidar', 'radar_lidar']
        self.dummy_parameter = nn.Parameter(tensor([0.]))
        #rank the modalities from 1.0 (best) to 0.4 (worst) for each scene based on domain knowledge.
        self.knowledge_base = {
            'city':     torch.FloatTensor([0.4, 0.8, 0.9, 0.5, 1.0, 0.7, 0.6]), #camera fusion > cameras > lidar > radar
            'rain':     torch.FloatTensor([0.9, 0.4, 0.5, 0.8, 0.6, 0.7, 1.0]), #radar+lidar > radar > lidar > camera+lidar > cameras
            'snow':     torch.FloatTensor([1.0, 0.4, 0.5, 0.8, 0.6, 0.7, 0.9]), #radar > radar+lidar > lidar > cameras
            'junction': torch.FloatTensor([0.4, 0.8, 0.9, 0.5, 1.0, 0.7, 0.6]), #camera fusion > cameras > lidar > radar
            'fog':      torch.FloatTensor([1.0, 0.4, 0.5, 0.8, 0.6, 0.7, 0.9]), #radar > radar+lidar > lidar > camera+lidar > cameras
            'tiny':     torch.FloatTensor([1.0, 0.4, 0.5, 0.8, 0.6, 0.7, 0.9]), #same as fog
            'motorway': torch.FloatTensor([0.4, 0.8, 0.9, 0.5, 1.0, 0.7, 0.6]), #camera fusion > cameras > lidar > radar
            'night':    torch.FloatTensor([0.8, 0.4, 0.5, 0.9, 0.6, 0.7, 1.0]), #radar+lidar > lidar > radar > cameras
            'rural':    torch.FloatTensor([0.4, 0.8, 0.9, 0.5, 1.0, 0.7, 0.6]), #camera fusion > cameras > lidar > radar
        }


    def forward(self, x, scene_type):
        return self.knowledge_base[scene_type]
