import matplotlib.pyplot as plt
import numpy as np

from imports import *


def pick_from_sides_inner(numbers, player):
    if len(numbers) <= 1:
        return [[0]], [[numbers[0] * player]]
    nsa = numbers[0]
    nsb = numbers[-1]
    ra, sa = pick_from_sides_inner(numbers[1:], -player)
    rb, sb = pick_from_sides_inner(numbers[:-1], -player)
    for i in range(len(ra)):
        ra[i].append(0)
        sa[i].append(nsa * player)
    for i in range(len(rb)):
        rb[i].append(1)
        sb[i].append(nsb * player)
    return ra + rb, sa + sb


def get_edge_states(numbers, actions):
    v = np.ones([actions.shape[0], actions.shape[1], 2])
    for i, al in enumerate(actions):
        leftc = 0
        rightc = len(numbers - 1)
        for j, a in enumerate(al):
            if a == 0:
                leftc += 1
            if a == 1:
                rightc -= 1
            v[i, j] = np.array([rightc, leftc])
    return np.stack(v, 0)


def get_game_states(numbers, actions):
    v = -np.ones(actions.shape[0], actions.shape[1], numbers.shape[0])
    for i, al in enumerate(actions):
        leftc = 0
        rightc = len(numbers - 1)
        for j, a in enumerate(al):
            if a == 0:
                leftc += 1
            if a == 1:
                rightc -= 1
            v[i, j, rightc:leftc] = numbers[rightc:leftc]
    return v


def pick_from_sides_array(numbers):
    r, s = pick_from_sides_inner(numbers, 1)
    return (np.array([[y for y in reversed(x)] for x in r]),
            np.array([[y for y in reversed(x)] for x in s]),
            numbers)


def pick_from_sides(length, count, max):
    numbers = np.random.randint(0, max, (count, length))
    numbers = set(tuple(row) for row in numbers)
    return [pick_from_sides_array(np.array(x)) for x in numbers]


def negamax(scores):
    split = len(scores) // 2
    if len(scores) == 1:
        return scores[0]
    r = max(negamax(-scores[:split]), negamax(-scores[split:]))
    return r


def negamaxfw(scores, player, f, depth, maxdepth, num_weaker_oracles, oraclemistakeninallafter=True):
    split = len(scores) // 2
    depthfromtop = maxdepth - depth
    if depthfromtop == 1:
        v = np.full((num_weaker_oracles + 1, 2), scores[0])
        f(v, depth + 1)
        return v
    lv = negamaxfw(scores[:split], -player, f, depth + 1, maxdepth,
                   num_weaker_oracles, oraclemistakeninallafter)
    rv = negamaxfw(scores[split:], -player, f, depth + 1, maxdepth,
                   num_weaker_oracles, oraclemistakeninallafter)
    mask = (lv[:, 1] < rv[:, 1]) if player == -1 else (lv[:, 0] > rv[:, 0])
    mask = np.stack([mask, mask], 1)
    r = np.where(mask, lv, rv)
    if depthfromtop - 1 < num_weaker_oracles + 1:
        idx = depthfromtop - 2 if oraclemistakeninallafter else 0
        vs = [lv[idx, 0], rv[idx, 0]]
        vassumed = np.average(vs)
        vtrue = np.min(vs)
        assert vtrue <= r[0, 0], f"{vassumed},{r}"
        if player == 1:
            r[depthfromtop - 1:, 0] = vassumed
            r[depthfromtop - 1:, 1] = vtrue
    # print(player, r, scores)
    f(r, depth + 1)
    return r


def negamaxf(scores, player, callback, depth, maxdepth, num_weaker_oracles, oracles_from,
             oraclemistakeninallafter=True):
    """

    Args:
        scores:
        player:
        callback: Function called with each node.
        depth:
        maxdepth:
        num_weaker_oracles:
        oraclemistakeninallafter:

    The opponent knows when weaker oracles are random and plays accordingly.
    Returns:

    """
    split = len(scores) // 2
    depthfromtop = maxdepth - depth
    if depthfromtop == 1:
        v = np.full((num_weaker_oracles + 1), scores[0])
        callback(v, depth + 1)
        return v
    lv = negamaxf(scores[:split], -player, callback, depth + 1, maxdepth,
                  num_weaker_oracles, oraclemistakeninallafter)
    rv = negamaxf(scores[split:], -player, callback, depth + 1, maxdepth,
                  num_weaker_oracles, oraclemistakeninallafter)
    mask = lv < rv if player == -1 else lv > rv
    r = np.where(mask, lv, rv)
    if depthfromtop - 1 > oracles_from and depthfromtop - 1 < num_weaker_oracles + 1 + oracles_from and player == 1:
        idx = depthfromtop - 2 if oraclemistakeninallafter else 0
        vs = [lv[idx], rv[idx]]
        vassumed = np.average(vs)
        assert np.all(vassumed <= r[:depthfromtop - 1]), f"{vassumed},{r}"
        r[depthfromtop - 1:] = vassumed
    # print(player, r, scores)
    callback(r, depth + 1)
    return r


def negamaxfbasic(scores, f, depth):
    split = len(scores) // 2
    if len(scores) == 1:
        f(scores[0], depth + 1)
        return scores[0]
    r = max(negamaxfbasic(-scores[:split], f, depth + 1), negamaxfbasic(-scores[split:], f, depth + 1))
    f(r, depth + 1)
    return r


def buildnegamaxtree(scores, maxdepth, num_weaker_oracles=0, oracles_from=0):
    """

    Args:
        scores: The action tree final layer scores. Assumes binary tree.

    Returns:
        The action tree branch representation of the negamax score before taking the action.
    """
    nodes = []

    def f(nodev, depth):
        nodes.append((nodev, depth))

    negamaxf(scores, 1, f, 0, maxdepth, num_weaker_oracles, oracles_from)
    maxd = nodes[0][1]
    numdescendantsmem = [2]
    branches = []
    for x in nodes:
        if maxd - x[1] == 0:
            branches.append([x[0]])
        else:
            if len(numdescendantsmem) < (maxd - x[1]):
                numdescendantsmem.append(numdescendantsmem[-1] * 2)
            for l in branches[-numdescendantsmem[maxd - x[1] - 1]:]:
                l.append(x[0])
    a = np.array([list(reversed(x)) for x in branches])
    # print(np.unique(a, return_counts=True), a.shape, a)
    return a


def groupinds(n, per):
    return np.arange(n).reshape(-1, per)


def group_pfs(d, inds, vs):
    def f(x):
        v = vs[x]
        return np.concatenate([v[0, :-d].flatten(), v[:, -d:].flatten()], 0)

    r = [f(x) for x in inds]
    return np.stack(r, 0)


def batch_group(d, inds, vs):
    def f(x):
        v = vs[:, :, x]
        return np.concatenate([np.reshape(v[:, :-d, 0], (v.shape[0], -1)),
                               np.reshape(v[:, -d:, :], (v.shape[0], -1))], 1)

    r = [f(x) for x in inds]
    return np.stack(r, 2)


def batch_truncate(d, inds, vs):
    def f(x):
        v = vs[:, :, x]
        return v[:, :-d, 0]

    r = [f(x) for x in inds]
    return np.stack(r, 2)


def truncate(actions, minmaxbs, depth):
    if depth == 0:
        return actions, minmaxbs
    inds = groupinds(minmaxbs.shape[2], 2 ** depth)
    leng = actions.shape[1]
    actionsn = batch_truncate(depth, inds, actions)[:, :leng - depth, :]
    minmaxbsn = batch_truncate(depth, inds, minmaxbs)[:, :leng - depth, :]
    return actionsn, minmaxbsn


def grouppfsscores(actions, scores, group_depth, truncate_depth):
    actions, scores = truncate(actions, scores, truncate_depth)
    if group_depth == 0:
        return np.concatenate([actions, scores], 1)
    inds = groupinds(scores.shape[2], 2 ** group_depth)
    leng = actions.shape[1]
    actionsn = batch_group(group_depth, inds, actions)[:, :leng - group_depth, :]
    scores = batch_group(group_depth, inds, scores)
    return np.concatenate([actionsn, scores], 1)


def grouppfs(actions, minmaxbs, group_depth, truncate_depth):
    actions, minmaxbs = truncate(actions, minmaxbs, truncate_depth)
    if group_depth == 0:
        return np.concatenate([actions,
                               np.reshape(np.transpose(minmaxbs, (0, 1, 3, 2)),
                                          [minmaxbs.shape[0], -1, minmaxbs.shape[2]])], 1)
    inds = groupinds(minmaxbs.shape[2], 2 ** group_depth)
    leng = actions.shape[1]
    actionsn = batch_group(group_depth, inds, actions)[:, :leng - group_depth, :]
    minmaxbsn = batch_group(group_depth, inds, minmaxbs)
    return np.concatenate([actionsn, minmaxbsn], 1)


def makepfs(leng, m, num_weaker_oracles, oracles_from, pathd="./data/pick_from_sides_data/d",
            pathl="./data/pick_from_sides_data/l"):
    d = []
    l = []
    vs = pick_from_sides(leng, 10000, m)
    for actions, scores, numbers in vs:
        scorescs = np.cumsum(scores, 1)
        minmaxbs = buildnegamaxtree(scorescs[:, -1], leng, num_weaker_oracles, oracles_from)
        d.append(np.concatenate([actions, scores, minmaxbs.reshape([minmaxbs.shape[0], -1])], 1).T)
        l.append(numbers)
    d = np.stack(d, 0)
    l = np.stack(l, 0)
    save(d, pathd)
    save(np.expand_dims(l, 2), pathl)
    return d, l

def get_views_pfs(data):
    r = []
    for p in range(int(np.log2(data.shape[0])) - 6):
        v = 2 ** p
        d = data.shape[2] // v
        if d == 0:
            print(f"{data.shape[2]} // {v} {data.shape[2] // v} != 0")
            continue
        for i in range(v):
            r.append(data[:, :, d * i: d * (i + 1)])

    return r

def select_bad_indices_pfs_seperate(numbers, minmaxes, scorescs, perc=7):
    l = len(numbers)
    s = numbers[0].shape[0]
    symmetrical = [x for x in range(l) if np.array_equal(numbers[x][s // 2:], numbers[x][:s // 2])]
    almost_all_one = [x for x in range(l) if np.any(np.unique(numbers[x], return_counts=True)[1] > s - 4)]
    # if winner is different from winner so far the signs differ so the product < 0
    interesting = interesting_playthroughs(minmaxes)
    interesting_scorescs = np.stack([scorescs[i, :, inds] for i, inds in enumerate(interesting)], 0)
    final_scores = interesting_scorescs[:, -1, :]
    ind = ((perc) * numbers.shape[0]) // 100
    top_draws = np.argsort(np.average(final_scores == 0, 1))[-ind:]
    non_final = interesting_scorescs[:, :-1, :]
    drama = np.sum(np.where(np.broadcast_to(np.expand_dims(final_scores, 1), non_final.shape) * non_final < 0,
                            np.sqrt(np.abs(interesting_scorescs[:, :-1, :])), 0),
                   2)
    average_drama = np.mean(drama, 1)
    bottom_drama = list(np.argsort(average_drama)[:ind].tolist())
    uncertainty = np.sum(
        np.square(
            np.linspace(0, final_scores, interesting_scorescs.shape[1], axis=1, endpoint=True) - interesting_scorescs),
        2)
    average_uncertainty = np.mean(uncertainty, 1)
    bottom_uncertainty = list(np.argsort(average_uncertainty)[:ind])
    ##NEED MORE IDEAS
    # Branching Factor for interesting
    # Move Effort score tradeoff
    # Killer Moves
    # Permanence
    # Lead Change
    # Stability
    # Coolness
    # Puzzle Potential
    r = [symmetrical, almost_all_one, top_draws, bottom_drama, bottom_uncertainty]
    n = 5
    empty = [i for i, x in enumerate(r) if len(x) <= n]
    if any(empty):
        print(f"some badness categories len <= {n} : {empty}")
    assert len(empty) < len(r), "all badness categories contain too few entries"
    return [np.unique(x).tolist() for x in r if len(x) > n]
