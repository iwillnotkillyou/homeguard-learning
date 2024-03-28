import numpy as np
import torch

from imports import *


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def info_nce_loss(query, positive_key, temperature):
    query, positive_key = normalize(query, positive_key)
    logits = query @ positive_key.transpose(-2, -1)
    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels)


class Dino_loss(torch.nn.Module):
    def __init__(self, dim, teacher_temp=1, student_temp=4, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros((1, dim)))

    def update(self, new_mean):
        with torch.no_grad():
            self.center = self.center_momentum * self.center + (1 - self.center_momentum) * new_mean.unsqueeze(0)

    def forward_one(self, teacher_output, student_output):
        student_lsm = F.log_softmax(student_output / self.student_temp, 1)
        teacher_sm = F.softmax((teacher_output - self.center) / self.teacher_temp, 1)
        return torch.mean(- teacher_sm * student_lsm)

    def forward_n_update(self, teacher_output, student_output):
        # print(student_output[0], teacher_output[0])
        # print(torch.equal(student_output[0], teacher_output[0]),torch.mean(torch.square(student_output[0] - teacher_output[0])))
        student_lsm = [F.log_softmax(x / self.student_temp, 1) for x in student_output]
        teacher_sm = [F.softmax((x - self.center) / self.teacher_temp, 1) if
                      x is not None else None for x in teacher_output]
        if self.training:
            with torch.no_grad():
                self.center = self.center_momentum * self.center + (1 - self.center_momentum) * mean(
                    [torch.mean(x, 0) for x in teacher_output if x is not None]).unsqueeze(0)
        vs = [[torch.mean(- t[1] * s[1], 0) for s in enumerate(student_lsm) if s[0] != t[0] and t[1] is not None] for t
              in
              enumerate(teacher_sm)]
        return flatten(vs)


def backward_dino(batch, device, teacher, student, config, criterion, muli, validate):
    data = batch[0]
    bs = data.shape[0]
    data = data[:bs // 2]
    p = config['label_coords_pos']
    action_ids = data[:, p[0]:p[1], :]
    maxd = 2
    loss = 0

    mean_teacher_output = 0
    views = []
    for i in range(maxd + 1):
        views = views + list((x, i) for x in get_views_indices(action_ids, data, i, bs))

    for i, s in enumerate(views):
        for j, t in enumerate(views):
            if i != j and t[1] < 2:
                s_in, t_in = get_view_from_indices(data, s[0]).to(device), get_view_from_indices(data, t[0]).to(device)
                teacher_output, student_output = (teacher.embedding(t_in).detach(), student.embedding(s_in))
                lv = criterion.forward_one(teacher_output, student_output)
                l = muli * lv
                if not validate:
                    l.backward()
                loss = loss + float(lv.item())
                mean_teacher_output = teacher_output + mean_teacher_output
                del teacher_output
                del student_output
                del s_in
                del t_in

    assert not isinstance(mean_teacher_output, type(0)), "no views found"
    criterion.update(mean_teacher_output)
    return np.array([loss])


def get_loss_dino(batch, device, teacher, student, config, criterion, epoch):
    data = batch[0].to(device)
    bs = data.shape[0]
    data = data[:bs // 8]
    p = config['label_coords_pos']
    action_ids = data[:, p[0]:p[1], :]
    maxd = 2

    def f(x, i):
        r = []
        for batch in list(x):
            r.append((teacher.embedding(batch).detach() if i == 0 else None, student.embedding(batch)))
        return r

    teacher_output, student_output = unzip(flatten([f(get_views(action_ids, data, i, bs), i)
                                                    for i in range(maxd + 1)]))
    losses_sep = [float(np.sqrt(student_output[0].shape[0])) * x for x in criterion(teacher_output, student_output)]
    loss = mean([torch.mean(x) for x in losses_sep])
    return loss, np.array([torch.mean(x).item() for x in losses_sep[:1]])


def get_seperate_losses_means(losses, num_highest=10):
    most_separate_losses_mean = [np.mean([x[i] for x in losses], (0,) + tuple(range(2, len(losses[0][i].shape))))
                                 for i in range(len(losses[0])) if losses[0][i] is not None]
    separate_losses_mean = [np.mean([x[i] for x in losses]) if losses[0][i] is not None else 0 for i in
                            range(len(losses[0]))]
    losses_mean = np.mean([np.mean([x[i] for x in losses]) for i in range(len(losses[0])) if losses[0][i] is not None])

    def f(x):
        if len(x) == 0:
            return ""
        x = np.squeeze(x)
        if len(x.shape) == 0:
            return ""
        s = np.argsort(x, 0)[::-1]
        vs = np.concatenate([np.expand_dims(x[s], 1), np.expand_dims(s, 1)], 1)[:num_highest]
        return [f"{float(vs[x, 0]):3.3},{int(vs[x, 1])}" for x in range(vs.shape[0])]

    most_separate_losses_mean = [f(x) for x in
                                 most_separate_losses_mean]
    return losses_mean, separate_losses_mean, most_separate_losses_mean


class HardBatchMiningTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def compute_distance_pairs(self, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(0)
        normcooefs = torch.sqrt(torch.sum(torch.square(inputs), 1)).unsqueeze(1)
        vs = inputs / normcooefs
        distance_matrix = torch.square(torch.cdist(vs, vs)).clamp(min=1e-12)

        distance_positive_pairs = torch.zeros(batch_size)
        distance_negative_pairs = torch.zeros(batch_size)
        for c in torch.unique(targets):
            idxs = torch.where(torch.eq(targets, c))
            idxs_neg = torch.where(torch.logical_not(torch.eq(targets, c)))
            idxs = idxs[0]
            idxs_neg = idxs_neg[0]
            for x in idxs:
                distance_positive_pairs[x] = torch.max(distance_matrix[x][idxs], 0).values
                distance_negative_pairs[x] = torch.min(distance_matrix[x][idxs_neg], 0).values

        distance_positive_pairs = distance_positive_pairs
        distance_negative_pairs = distance_negative_pairs

        return distance_positive_pairs, distance_negative_pairs

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        distance_positive_pairs, distance_negative_pairs = self.compute_distance_pairs(inputs, targets)
        distance_positive_pairs = distance_positive_pairs.to(inputs.device)
        distance_negative_pairs = distance_negative_pairs.to(inputs.device)
        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        # This is done already, no need to change anything.
        y = torch.ones_like(distance_negative_pairs)
        # one in y indicates, that the first input should be ranked higher than the second input, which is true for all the samples
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)


def triplet_loss_kmeans(l, prediction, label, device):
    bs = label.shape[0]

    def f(inp, clusters):
        km = sklearn.cluster.KMeans(clusters, n_init=10)
        km.fit(inp)
        return km.predict(inp), km.score(inp)

    return l(prediction, torch.from_numpy(max([f(label.cpu().numpy(), x)
                                               for x in range(2, min(bs // 2, 16), 2)],
                                              key=lambda x: x[1])[0]).to(device))


def get_triplet_loss(data1, data2, label, device, model):
    data = torch.concatenate([data1, data2], 0)
    label = torch.concatenate([label, label], 0)
    prediction = model.embedding(data)
    l = HardBatchMiningTripletLoss()
    return l(prediction, label[:, 0])


def get_completion_loss(data, label1, label2, model):
    prediction = model(data, label1)
    loss_total = torch.mean(torch.square(prediction - label2), (0, 2))
    return loss_total


def get_loss_multitask_with_triplet(batch, device, model, config, epoch):
    p = config['label_coords_pos']
    label = batch[1].to(device)
    data = shuffle_bottom_all_same(batch[0]).to(device)
    s = data.shape[2] // 2
    data1, data2 = data[..., :s], data[..., s:]
    if config["max_epochs"] > config["contrastive_after"]:
        a = config["perc_contrastive"] * (
                max(0, epoch - config["contrastive_after"]) / (config["max_epochs"] - config["contrastive_after"]))
    else:
        a = 0
    cl = get_completion_loss(data1, data2[:, p[0]:p[1], :], data2[:, p[1]:, :], model).unsqueeze(1)
    clm = torch.mean(cl)
    tl = get_triplet_loss(data1, data2, label, device, model) if a > 0 else clm * 0
    loss_total = a * torch.mean(tl) + (1 - a) * clm
    return loss_total, [tl.detach().cpu().numpy(), cl.detach().cpu().numpy()]


def backward_loss_multitask_with_dino(batch, device, model, config, criterion, epoch, validate):
    p = config['label_coords_pos']
    data = batch[0]
    s = data.shape[2] // 2
    inds = np.arange(data.shape[2])
    np.random.shuffle(inds)
    data1, data2 = data[..., inds[:s]].to(device), data[..., inds[s:]].to(device)
    if config["max_epochs"] > config["contrastive_after"]:
        a = config["perc_contrastive"] * (
                max(0, epoch - config["contrastive_after"] + 1) / (
                config["max_epochs"] - config["contrastive_after"] + 1))
    else:
        a = 0

    # print(np.average([1 for mp, tp in zip(model.auto_encoder.parameters(), teacher.parameters()) if torch.equal(mp.data, tp.data)]))
    completion_loss = get_completion_loss(data1, data2[:, p[0]:p[1], :], data2[:, p[1]:, :],
                                          model.auto_encoder).unsqueeze(1)
    if not validate:
        torch.mean(completion_loss).backward()
    cl = completion_loss.detach().cpu().numpy()
    del completion_loss
    contrastive_loss = backward_dino(batch, device, model.teacher, model.dino, config,
                                     criterion, a, validate) if a > 0 else None
    loss_total = np.mean(cl) if contrastive_loss is None else a * np.mean(contrastive_loss) + (1 - a) * np.mean(cl)
    return loss_total, [contrastive_loss, cl]


def get_loss_multitask_with_dino(batch, device, model, config, criterion, epoch):
    p = config['label_coords_pos']
    data = batch[0]
    s = data.shape[2] // 2
    inds = np.arange(data.shape[2])
    np.random.shuffle(inds)
    data1, data2 = data[..., inds[:s]].to(device), data[..., inds[s:]].to(device)
    if config["max_epochs"] > config["contrastive_after"]:
        a = config["perc_contrastive"] * (
                max(0, epoch - config["contrastive_after"] + 1) / (
                config["max_epochs"] - config["contrastive_after"] + 1))
    else:
        a = 0

    # print(np.average([1 for mp, tp in zip(model.auto_encoder.parameters(), teacher.parameters()) if torch.equal(mp.data, tp.data)]))
    completion_loss = get_completion_loss(data1, data2[:, p[0]:p[1], :], data2[:, p[1]:, :],
                                          model.auto_encoder).unsqueeze(1)
    completion_loss_mean = torch.mean(completion_loss)
    contrastive_loss_mean, contrastive_loss = get_loss_dino(batch, device, model.teacher, model.dino, config,
                                                            criterion,
                                                            epoch) if a > 0 else [completion_loss_mean * 0, None]
    loss_total = a * torch.mean(contrastive_loss_mean) + (1 - a) * completion_loss_mean
    return loss_total, [contrastive_loss, completion_loss.detach().cpu().numpy()]
