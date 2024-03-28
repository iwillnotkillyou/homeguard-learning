from imports import *
from modules import *
from itertools import chain


def copy_params(m1, m2):
    for m1p, m2p in zip(m1.parameters(), m2.parameters()):
        m1p.data = torch.clone(m2p.data.detach())


class TransformerAutoEncoder(nn.Module):
    def __init__(self, backbone, target_in_channels, target_channels, dropouts):
        super().__init__()
        self.target_in_channels = target_in_channels
        self.target_channels = target_channels
        self.backbone = backbone
        self.latents = backbone.outsize
        self.partial_latents = backbone.partial_outsize
        self.decoder = make_MLP([target_in_channels + self.latents, self.partial_latents],
                                lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.decoder2 = MultiHeadAttention(self.partial_latents, 4)
        self.decoder3 = MultiHeadAttention(self.partial_latents, 4)
        self.final = nn.Conv1d(self.partial_latents, target_channels, 1)
        self.dropouts = dropouts

    def embedding(self, x):
        return self.backbone(x)

    def freeze_first(self):
        for x in chain(self.encoder[0].parameters(), self.encoder[1].parameters()):
            x.requires_grad = False

    def unfreeze(self):
        for x in self.parameters():
            x.requires_grad = False

    def forward(self, x, targets):
        x, partial_x, partial_x2 = self.backbone.forward_w_partial_embeddings(x)
        x = nn.functional.dropout(x, self.dropouts[0])
        partial_x = nn.functional.dropout1d(partial_x, self.dropouts[1])
        partial_x2 = nn.functional.dropout1d(partial_x2, self.dropouts[2])
        embed = broadcast_unsqueeze(x, 2, targets.shape[2])
        x = self.decoder(torch.cat([embed, targets], 1))
        x = self.decoder2.multi_forward(x, partial_x, partial_x)
        x = self.decoder2.multi_forward(x, partial_x2, partial_x2)
        return self.final(x)


class PointNetAutoEncoder(nn.Module):
    def __init__(self, backbone, target_in_channels, target_channels, decoutsize):
        super().__init__()
        self.target_in_channels = target_in_channels
        self.target_channels = target_channels
        self.backbone = backbone
        self.decoutsize = decoutsize
        self.latents = backbone.outsize
        self.decoder = make_MLP([target_in_channels + self.latents, self.decoutsize],
                                lambda x, y: SelfAttention(x, y))
        self.final = nn.Conv1d(self.decoutsize + self.latents, target_channels, 1)

    def embedding(self, x):
        return self.backbone(x)

    def freeze_first(self):
        for x in chain(self.encoder[0].parameters(), self.encoder[1].parameters()):
            x.requires_grad = False

    def unfreeze(self):
        for x in self.parameters():
            x.requires_grad = False

    def forward(self, x, targets):
        x = self.backbone.forward_w_partial_embeddings(x)
        if isinstance(x, tuple):
            x = x[0]
        embed = broadcast_unsqueeze(x, 2, targets.shape[2])
        x = self.decoder(torch.cat([embed, targets], 1))
        return self.final(torch.cat([x, embed], 1))


def max_broadcast(x):
    return torch.max(x, 2, keepdim=True)[0].broadcast_to(x.shape[0], x.shape[1], x.shape[2])


class PointNetEmbedderSmall(nn.Module):
    def __init__(self, in_channels, outsize, sizes):
        super().__init__()
        self.in_channels = in_channels
        self.sizes = sizes
        self.encoder1 = make_MLP([in_channels, sizes[0]],
                                 lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder2 = make_MLP([sizes[0] + in_channels, sizes[1]],
                                 lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.final = nn.Linear(sizes[1], outsize)
        self.outsize = outsize

    def forward(self, x: torch.Tensor):
        x1 = self.encoder1(x)
        x = torch.cat([x, max_broadcast(x1)], 1)
        x = self.encoder2(x)
        x = x.max(2)[0]
        return self.final(x)


class TransformerEmbedder(nn.Module):
    def __init__(self, in_channels, outsize, sizes, dropouts):
        super().__init__()
        self.in_channels = in_channels
        self.sizes = sizes
        self.encoder0 = MultiHeadAttention(in_channels)
        self.encoder1 = ConvWithReluNBatchNorm(in_channels, sizes[0])
        self.dropouts = dropouts
        self.encoder2 = MultiHeadAttention(sizes[0], 8, dropout=self.dropouts[0])
        self.encoder3 = MultiHeadAttention(sizes[0], 8, dropout=self.dropouts[1])
        self.final = nn.Linear(sizes[0], outsize)
        self.outsize = outsize
        self.partial_outsize = sizes[0]

    def forward_w_partial_embeddings(self, x: torch.Tensor):
        x = self.encoder0(x)
        x = self.encoder1(x)
        x2 = self.encoder2(x)
        x3 = self.encoder3(x2)
        x = x3
        xmax = torch.max(x, 2)[0]
        #xmin = torch.max(x, 2)[0]
        # xsmax = torch.sum(torch.nn.functional.softmax(x, 2) * x, 2)
        # xsmaxt2 = torch.sum(torch.nn.functional.softmax(x / 2, 2) * x, 2)
        # x = torch.cat([xmax, xsmax, xsmaxt2], 1)
        #x = torch.cat([xmax, xmin])
        x = xmax
        return self.final(x), x2, x3

    def forward(self, x: torch.Tensor):
        return self.forward_w_partial_embeddings(x)[0]


class PointNetEmbedder(nn.Module):
    def __init__(self, in_channels, outsize, sizes):
        super().__init__()
        self.in_channels = in_channels
        self.sizes = sizes
        self.encoder11 = make_MLP([in_channels, sizes[0]],
                                  lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder12 = make_MLP([sizes[0], sizes[0]],
                                  lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder21 = make_MLP([sizes[0] + in_channels, sizes[1]],
                                  lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder22 = make_MLP([sizes[1] + in_channels, sizes[1]],
                                  lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder3 = make_MLP([sizes[1] + sizes[0], sizes[1] * 2],
                                 lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.encoder4 = make_MLP([sizes[1] * 2 + sizes[1], sizes[1] * 2],
                                 lambda x, y: ConvWithReluNBatchNorm(x, y))
        self.final = nn.Linear(sizes[1] * 2, outsize)
        self.outsize = outsize

    def forward(self, x: torch.Tensor):
        x11 = self.encoder11(x)
        x12 = self.encoder12(x11)
        x = torch.cat([x, max_broadcast(x12)], 1)
        x21 = self.encoder21(x)
        x22 = self.encoder21(x)
        x = torch.cat([x11, max_broadcast(x22)], 1)
        x = self.encoder3(x)
        x = torch.cat([x21, max_broadcast(x)], 1)
        x = self.encoder4(x)
        x = x.max(2)[0]
        return self.final(x)


class PointNetDino(nn.Module):
    def __init__(self, backbone, hss, dim):
        super().__init__()
        self.backbone = backbone
        self.linears = make_MLP_with_last((backbone.outsize,) + tuple(hss) + (dim,), lambda x, y: nn.Linear(x, y),
                                          lambda x, y: nn.utils.parametrizations.weight_norm(nn.Linear(x, y)))
        self.dim = dim
        self.hss = hss

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, tuple):
            x = x[0]
        return self.linears(x)

    def embedding(self, x):
        return self.forward(x)


class DinoCompletionModuleList(nn.Module):
    def __init__(self, auto_encoder, dino, teacher):
        super().__init__()
        self.auto_encoder = auto_encoder
        self.dino = dino
        self.teacher = teacher

    def embedding(self, x):
        return self.dino(x) if not isinstance(self.dino, nn.Identity) else self.auto_encoder.embedding(x)
