from imports import *


def select_bad_indices_homeguard_seperate(numbers, minmaxes, scorescs, perc=15):
    l = len(numbers)
    s = numbers[0].shape[0]
    symmetrical = [x for x in range(l) if np.array_equal(numbers[x][s // 2:], numbers[x][:s // 2])]
    # if winner is different from winner so far the signs differ so the product < 0
    interesting = interesting_playthroughs(minmaxes)
    interesting_scorescs = np.stack([scorescs[i, :, inds] for i, inds in enumerate(interesting)], 0)
    final_scores = interesting_scorescs[:, -1, :]
    ind = ((perc) * numbers.shape[0]) // 100
    top_draws = np.argsort(np.average(final_scores == 0, 1))[-ind:]
    non_final = interesting_scorescs[:, :-1, :]
    drama = np.mean(np.where(np.broadcast_to(np.expand_dims(final_scores, 1), non_final.shape) * non_final < 0,
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
    r = [symmetrical, #top_draws,
         bottom_drama, bottom_uncertainty]
    names = ["symmetrical", #"top_draws",
             "bottom_drama", "bottom_uncertainty"]
    n = 10
    empty = [i for i, x in enumerate(r) if np.unique(x).shape[0] <= n]
    if any(empty):
        print(f"some badness categories len <= {n} : {empty}")
    assert len(empty) < len(r), f"all badness categories contain too few entries {[np.unique(x).shape[0] for x in r]}"
    return unzip([(np.unique(x).tolist(), name) for x, name in zip(r, names) if len(x) > n])
