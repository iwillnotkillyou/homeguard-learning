import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MySelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None):
        super().__init__()
        inter_channels = in_channels \
            if inter_channels is None \
            else inter_channels
        self.out_channels = out_channels
        self.query = ConvWithReluNBatchNorm(in_channels, inter_channels)
        self.key = ConvWithReluNBatchNorm(in_channels, inter_channels)
        self.value = ConvWithReluNBatchNorm(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries.transpose(1, 2), keys) / (self.out_channels ** 0.5)
        attention = self.softmax(scores)
        weighted = (torch.bmm(attention, values.transpose(1, 2))
                    .transpose(1, 2))
        assert weighted.shape[2] == x.shape[2]
        return weighted


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1, dropout=0):
        super().__init__()
        self.layer = nn.MultiheadAttention(in_channels, num_heads, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = torch.nn.ReLU()

    def multi_forward(self, q, k, v):
        x = self.layer.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), need_weights=False)[0]
        x = self.layer.forward(x, x, x, need_weights=False)[0]
        x = x.transpose(1, 2)
        return self.relu(self.bn(x))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer.forward(x, x, x, need_weights=False)[0]
        x = x.transpose(1, 2)
        return self.relu(self.bn(x))


class MultiHeadAttentionWithConvReluNBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, dropout=0):
        super().__init__()
        self.layer = nn.MultiheadAttention(in_channels, num_heads, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_channels)
        self.conv = ConvWithReluNBatchNorm(in_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def multi_forward(self, q, k, v):
        x = self.layer.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), need_weights=False)[0]
        x = self.layer.forward(x, x, x, need_weights=False)[0]
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.relu(self.bn(x))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer.forward(x, x, x, need_weights=False)[0]
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.relu(self.bn(x))


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None, dropout_p=0.0):
        super().__init__()
        inter_channels = ((min(in_channels, out_channels))
                          if inter_channels is None
                          else inter_channels)
        self.dropout_p = dropout_p
        self.out_channels = out_channels
        self.query = nn.Conv1d(in_channels, inter_channels, 1)
        self.key = nn.Conv1d(in_channels, inter_channels, 1)
        self.value = nn.Conv1d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        return torch.nn.functional.scaled_dot_product_attention(queries,
                                                                keys,
                                                                values,
                                                                dropout_p=self.dropout_p)


class ConvWithReluNBatchNorm(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1):
        super().__init__()
        self.nn = torch.nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.nn(x)
        return self.relu(self.bn(x))


def make_MLP(channels, layer):
    ls = []
    for x in range(len(channels) - 1):
        ls.append(layer(channels[x], channels[x + 1]))
    return torch.nn.Sequential(*ls)


def make_MLP_with_last(channels, layer, layerLast):
    return make_MLP_index(channels, lambda x, y, i: layerLast(x, y) if i == len(channels) - 2 else layer(x, y))


def make_MLP_index(channels, layer):
    ls = []
    for x in range(len(channels) - 1):
        ls.append(layer(channels[x], channels[x + 1], x))
    return torch.nn.Sequential(*ls)


class LinearWithReluNBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.nn(x)))


class TNet(nn.Module):
    def __init__(self, k, base):
        super().__init__()
        self.convs = torch.nn.Sequential(ConvWithReluNBatchNorm(k, base),
                                         ConvWithReluNBatchNorm(base, base * 2),
                                         ConvWithReluNBatchNorm(base * 2, 1024))
        self.linears = torch.nn.Sequential(LinearWithReluNBatchNorm(1024, base * 8),
                                           LinearWithReluNBatchNorm(base * 8, base * 4),
                                           torch.nn.Linear(base * 4, k * k))
        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]
        x = self.convs(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.linears(x)
        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x
