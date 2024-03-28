import time

import numpy as np
import torch

from imports import *


def get_order_elimination(distance_matrix, bad_known):
    all = np.arange(distance_matrix.shape[0])
    assert np.unique(bad_known).shape[0] == np.array(bad_known).shape[
        0], f"{np.unique(bad_known).shape[0]} != {np.array(bad_known).shape[0]}"
    non_eliminated = np.copy(all)
    order = []
    while True:
        for center in list(bad_known):
            if non_eliminated.shape[0] == 0:
                return order
            next_del_ind = np.argmin(distance_matrix[center][non_eliminated])
            actual_ind = non_eliminated[next_del_ind]
            non_eliminated = np.delete(non_eliminated, next_del_ind)
            order.append(actual_ind)


def seperate(order, bad_indices_separate):
    precisions = []
    for i in range(len(bad_indices_separate)):
        bi = bad_indices_separate[i]
        precisions.append(np.cumsum(np.where(np.isin(order, bi), 1, 0)) / bi.shape[0])
    precisionsm = np.mean(precisions, 0)
    return np.mean(precisionsm[int(0.75 * precisionsm.shape[0]):int(0.875 * precisionsm.shape[0])]), precisions


def get_order_mean(distance_matrix, bad_known):
    order = np.argsort(np.mean(distance_matrix[bad_known], 0))
    return order


def interesting_playthroughs(minmaxes):
    ms = minmaxes
    if len(ms.shape) > 3:
        ms = ms[:, :, :, 0]
    inds = []
    for m in list(ms):
        inds.append(np.argsort(m[m.shape[0] // 2, :])[:m.shape[1] // 2])
    return inds


def get_borders_tensor(x):
    return torch.where(torch.prod(torch.where(x == torch.roll(x, 1, 1), 1, 0), 0) == 0)[0]

def get_borders(x):
    return np.where(np.prod(np.where(x == np.roll(x, 1, 1), 1, 0), 0) == 0)[0]

def get_regions(borders, length):
    return zip(borders.tolist(), np.roll(borders, -1, 0)[:-1].tolist() + [length])

def get_regions_torch(borders, length):
    return zip(borders.cpu().numpy().tolist(), torch.roll(borders, -1, 0)[:-1].cpu().numpy().tolist() + [length])


def call_on_views(sorted_actions_ids, sorted_data, depth, out_bs, func):
    r = []
    vs = sorted_actions_ids[:sorted_actions_ids.shape[0], :depth + 1, :sorted_actions_ids.shape[2]]
    for i in range(vs.shape[0]):
        borders = get_borders_tensor(vs[i])
        for start, end in get_regions_torch(borders, sorted_data.shape[2]):
            r.append(sorted_data[i, :, start:end])
            if len(r) == out_bs:
                d = max([x.shape[1] for x in r])
                r = [torch.nn.functional.pad(x, (0, d - x.shape[1], 0, 0), "constant", 0) for x in r]
                r = torch.stack(r, 0)
                func(r)

def get_view_from_indices(data, inds):
    r = [data[x[0], :, x[1]:x[2]] for x in inds]
    d = max([x.shape[1] for x in r])
    r_inter = [torch.nn.functional.pad(x, (0, d - x.shape[1], 0, 0), "constant", 0) for x in r]
    r = torch.stack(r_inter, 0)
    return r

def get_views_indices(sorted_actions_ids, sorted_data, depth, out_bs):
    r = []
    r_inner = []
    vs = sorted_actions_ids[:sorted_actions_ids.shape[0], :depth + 1, :sorted_actions_ids.shape[2]]
    for i in range(vs.shape[0]):
        borders = get_borders_tensor(vs[i])
        for start, end in get_regions_torch(borders, sorted_data.shape[2]):
            r_inner.append((i, start, end))
            if len(r_inner) == out_bs:
                r.append(list(r_inner))
                r_inner = []

    return r

def get_views(sorted_actions_ids, sorted_data, depth, out_bs):
    r = []
    vs = sorted_actions_ids[:sorted_actions_ids.shape[0], :depth + 1, :sorted_actions_ids.shape[2]]
    for i in range(vs.shape[0]):
        borders = get_borders_tensor(vs[i])
        for start, end in get_regions_torch(borders, sorted_data.shape[2]):
            r.append(sorted_data[i, :, start:end])
    remainder = len(r) % out_bs
    if remainder != 0:
        r = r[:-remainder]
    d = max([x.shape[1] for x in r])
    r = [torch.nn.functional.pad(x, (0, d - x.shape[1], 0, 0), "constant", 0) for x in r]
    r = torch.stack(r, 0)
    return r.view(-1, out_bs, r.shape[1], r.shape[2])

def group(data, sorted_actions_ids, amount):
    bs = sorted_actions_ids.shape[0]
    game_length = sorted_actions_ids.shape[1]
    num_branches = sorted_actions_ids.shape[2]
    from_start = game_length - amount
    if amount > 0:
        descendants = []
        max_siblings = 0
        vs = sorted_actions_ids[:, :from_start, :]
        for i in range(bs):
            borders = get_borders(vs[i])
            reg_sizes = [end - start for start, end in get_regions(borders, num_branches)]
            descendants.append(reg_sizes)
            #print(f"min {np.min(reg_sizes)}", f"max {np.max(reg_sizes)}", np.unique(reg_sizes))
            max_siblings = max(max_siblings, borders.shape[0])
        counts = np.unique(flatten(descendants))
        counts = np.sort(counts)
        counts_cs = np.cumsum(counts)
        ceiling_descendants = np.where(counts_cs > counts[-1] * 0.95)[0][0]
        grouped = np.zeros([bs, from_start + ceiling_descendants * amount, max_siblings + 1], dtype=data.dtype)
        for i in range(bs):
            borders = get_borders(vs[i])
            for j, (start, end) in enumerate(get_regions(borders, num_branches)):
                if (end - start) > ceiling_descendants:
                    end = start + ceiling_descendants
                grouped[i, :from_start, j] = data[i, :from_start, start]
                grouped[i, from_start: from_start + (end - start) * amount, j] = data[i, from_start:, start:end].flatten()
        return grouped
    else:
        return data

def sort(action_ids, a):
    for i in range(action_ids.shape[0]):
        inds = np.lexsort(action_ids[i, ::-1])
        a[i] = a[i][:, inds]
        action_ids[i] = action_ids[i][:, inds]

def calculate_minmax(sorted_actions_ids, sorted_scores_cumsums):
    minmaxes = np.copy(sorted_scores_cumsums)
    bs = sorted_actions_ids.shape[0]
    game_length = sorted_actions_ids.shape[1]
    num_branches = sorted_actions_ids.shape[2]
    print(np.unique(sorted_scores_cumsums[:, -1, :]))
    for j, x in enumerate(reversed(range(0, game_length - 1))):
        vs = sorted_actions_ids[:, :x + 1, :]
        if (x + 1) // 2 == 0:
            minmaxes[:, x, :] = -np.inf
        else:
            minmaxes[:, x, :] = np.inf
        assert np.isfinite(minmaxes[:, x + 1, :]).all(), str(
            np.where(np.logical_not(np.isfinite(minmaxes)[:, x + 1, :])))
        for i in range(bs):
            borders = get_borders(vs[i])
            print(x, i, np.unique(minmaxes[i, x + 1, :]))
            for start, end in get_regions(borders, num_branches):
                if (x + 1) // 2 == 0:
                    m = np.max(minmaxes[i, x + 1, start:end])
                else:
                    m = np.min(minmaxes[i, x + 1, start:end])
                minmaxes[i, x, start:end] = m
            minmaxes[i, x, borders[-1]:] = m

    print(np.where(np.logical_not(np.isfinite(minmaxes))))
    assert np.isfinite(minmaxes).all()
    return minmaxes
