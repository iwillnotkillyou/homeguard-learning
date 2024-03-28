import os

import matplotlib.pyplot as plt
import numpy as np

from imports import *


def averageauc(embeds, bad_indices, mapsf=None, fig_path=None, use_mean=False, known=10):
    (bad_indices, bad_indices_separate, bad_cat_names) = bad_indices
    embeds = embeds.unsqueeze(0)
    normcooefs = torch.sqrt(torch.sum(torch.square(embeds), 2)).unsqueeze(2).clamp(min=1e-12)
    vs = embeds / normcooefs
    bad_indices = np.array(bad_indices)
    assert (np.unique(bad_indices).shape[0] == bad_indices.shape[0])
    distance_matrix = torch.square(torch.cdist(vs, vs)).clamp(min=1e-12).squeeze().cpu().numpy()
    aucs = []
    precisionsl = []

    # print(10/bad_indices.shape[0])
    def f(bi, inds, i, aucs):
        order_f = get_order_mean if use_mean else get_order_elimination
        random_vs = []

        def f(precisions):
            return np.stack(precisions +
                            [np.mean(precisions, 0)] +
                            [np.mean(allnotat(precisions, inds), 0)
                             if len(precisions) > len(inds) > 0
                             else np.zeros(precisions[0].shape[0])], 0)

        for x in range(100):
            selection = np.random.choice(bi.shape[0], known, replace=False)
            order = order_f(torch.rand(distance_matrix.shape), selection)
            random_auc, random_precisions = seperate(order, bad_indices_separate)
            random_vs.append(f(random_precisions))
        vs = []
        for x in range(100):
            selection = np.random.choice(bi.shape[0], known, replace=False)
            order = order_f(distance_matrix, bi[selection])
            auc, precisions = seperate(order, bad_indices_separate)
            aucs.append(auc)
            vs.append(f(precisions))
            if i == 0 and x == 0 and mapsf is not None:
                mapsf(order[-len(order) // 8:])
        print(i)
        return np.mean(vs, 0), np.mean(random_vs, 0)

    for i, inds in enumerate(powerset(range(len(bad_indices_separate)))):
        if len(inds) == 0:
            continue
        bi = np.unique(np.concatenate(allat(bad_indices_separate, inds), 0))
        if bi.shape[0] < known:
            print(f"{inds} too few members")
            continue
        precisions, random_precisions = f(bi, inds, i, aucs
        if len(inds) == len(bad_indices_separate) else [])
        precisionsl.append((precisions, random_precisions, inds))

    def f2(precisionsl, random_precisionsl, names):
        for precisions, random_precisions, name in zip(precisionsl, random_precisionsl, names):
            print(name, os.path.dirname(name))
            os.makedirs(os.path.dirname(name), exist_ok=True)
            print(precisions.shape)
            x = np.linspace(0, 1, precisions.shape[1])
            plt.plot(x, x, color='blue')
            for i in range(precisions.shape[0] - 2):
                plt.plot(x, precisions[i], label=f"{bad_cat_names[i]}")
            plt.legend()
            plt.savefig(name + "_seperate.png")
            plt.clf()

            def f(vs, random_vs, name):
                plt.plot(x, random_vs, color='blue', label="random")
                plt.plot(x, vs, color='red', label="proper")
                plt.fill_between(x, vs, random_vs, where=np.logical_and(np.logical_and(x > 0.75, x < 0.875),
                                                                        vs > random_vs), color='green',
                                 alpha=0.3,
                                 interpolate=True)
                plt.legend()
                plt.savefig(name)
                plt.clf()

            f(precisions[-1], random_precisions[-1], name + "_other.png")
            f(precisions[-2], random_precisions[-2], name + "_all.png")

    if fig_path is None:
        fig_path = "figs/average"

    precisionsl, random_precisionsl, indsl = unzip3(precisionsl)
    names = [fig_path + "/f" + ("-".join([str(y) for y in x])
                                if len(x) < len(bad_indices_separate)
                                else "all") for x in indsl]
    f2(precisionsl, random_precisionsl, names)
    return np.mean(aucs)


def print_info(embeds, bad_indicest, loss, params, rs, mapsf, save_fig):
    embedsus = embeds.unsqueeze(0)
    normcooefs = torch.sqrt(torch.sum(torch.square(embedsus), 2)).unsqueeze(2).clamp(min=1e-12)
    vs = embedsus / normcooefs
    bad_indices = np.array(bad_indicest[0])
    distance_matrix = torch.square(torch.cdist(vs, vs)).clamp(min=1e-12).squeeze().cpu().numpy()
    bad_dist = np.mean(distance_matrix[bad_indices]).item()
    ##we want average_dist << bad_dist
    average_dist = np.mean(distance_matrix).item()
    last_bad = []
    count_bad_all = len(bad_indices)
    perc_bad_lowest50 = []
    perc_bad_lowest25 = []
    perc_bad_highest5 = []
    for _ in range(10):
        selection = np.random.choice(count_bad_all, 10, replace=False)
        order = np.argsort(np.mean(distance_matrix[selection], 0))
        # print(count_bad_all)
        # perc_bad_75 = len(set(order[0:distance_matrix.shape[0] // 4 * 3]).intersection(badindss)) / count_bad_all
        # print(perc_bad_75)
        perc_bad_lowest50.append(
            np.intersect1d(order[:distance_matrix.shape[0] // 2], bad_indices).shape[0] / count_bad_all)
        perc_bad_lowest25.append(
            np.intersect1d(order[:distance_matrix.shape[0] // 4], bad_indices).shape[0] / count_bad_all)
        perc_bad_highest5.append(
            np.intersect1d(order[(distance_matrix.shape[0] // 20) * 19:], bad_indices).shape[0] / count_bad_all)
        last_bad.append(
            np.max(order[np.intersect1d(order, bad_indices, return_indices=True)[1]]) /
            distance_matrix.shape[0])

    avg_auc = averageauc(embeds, bad_indicest, mapsf if False else lambda x: x,
                         fig_path=f"figs/average/{'-'.join([str(x) for x in params])}" if save_fig else None)
    # perc_bad_lowest = len(set(order[0:len(bad_val_inds)]).intersection(badindss)) / len(bad_val_inds)
    rs.append((params, avg_auc,
               (loss, avg_auc, np.mean(perc_bad_lowest50), np.mean(perc_bad_lowest25), np.mean(perc_bad_highest5),
                np.mean(last_bad),
                (bad_dist - average_dist) / average_dist,
                average_dist / bad_dist, bad_dist, average_dist)))
    print(rs)
    print(sorted(rs, key=lambda x: x[1], reverse= True))
    rsa = np.array([x[2] for x in sorted(rs, key=lambda x: x[1])])
    names = ["loss", "avg_auc", "perc_bad_lowest50", "perc_bad_lowest25", "perc_bad_highest5", "last_bad",
             "(bad_dist - average_dist)/(average_dist)",
             "average_dist/bad_dist", "bad_dot", "average_dot"]
    names1 = eqelsenone(names, {"loss", "perc_bad_lowest50", "perc_bad_lowest25"})
    names2 = eqelsenone(names, {"perc_bad_highest5", "last_bad", "avg_auc"})
    names3 = eqelsenone(names, {"bad_dot", "average_dot"})
    print_n_log(str(describe(rsa, names1)))
    print_n_log(str(describe(rsa, names2)))
    print_n_log(str(describe(rsa, names3)))


def shuffle_bottom_all_same(x):
    inds = np.arange(x.shape[2])
    np.random.shuffle(inds)
    return x[:, :, inds]


def shuffle_bottom(x):
    r = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            inds = np.arange(x.shape[2])
            np.random.shuffle(inds)
            v = x[i, j, inds]
            r[i, j, :] = v
    return r
