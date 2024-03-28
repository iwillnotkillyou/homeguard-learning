from imports import *


def loadDataDescStrings(path):
    v = open(path).read()
    v1 = str.split(v, "----\n")[-1]
    r = str.split(v1, "\n\n\n")
    return r


class IdentityTransform:
    """Transform class that does nothing"""

    def __call__(self, images):
        return images


def loadDataDescClass(strings, ind):
    return np.asarray(list(x.split("\n\n")[1].split(";")[ind] for x in strings))


def loadDataDescMapHeiOnly(strings):
    orders = np.asarray(
        [[[[int(x) if str.isnumeric(x) else 0] for y in x.split("\n\n")[0].split("\n")] for x in strings]])
    return orders


def loadDataDescMap(strings):
    strings = [x.split("\n\n")[0] for x in strings]
    orders = np.asarray([[[ord(x) for x in y] for y in x.split("\n")] for x in strings])
    unique = np.unique(orders)
    for i in range(len(unique)):
        orders = np.where(orders == unique[i], i, orders)
    return orders, np.array(strings)


def load3dArray(path):
    f = open(path, 'rb')
    xd = int.from_bytes(f.read(4), byteorder="little", signed=False)
    yd = int.from_bytes(f.read(4), byteorder="little", signed=False)
    zd = int.from_bytes(f.read(4), byteorder="little", signed=False)
    data = np.fromfile(f, dtype=np.dtype(np.single).newbyteorder("<"))
    data = data.reshape([xd, yd, zd])
    return data


def as_one_hot(data):
    vs = []
    distinct = int(np.max(data))
    for dim in range(data.shape[1]):
        for c in range(distinct):
            vs.append(np.where(data[:, dim, :] == c, np.ones(1, dtype=np.single),
                               np.zeros(1, dtype=np.single)))
    return np.stack(vs, 1)


def loadBranches(path, pathInds, onehot=True):
    data = load3dArray(path)
    data = data.transpose((0, 2, 1))
    data = data[:, 1:, :]
    vs = data
    if pathInds is not None:
        lastActionIndex = load3dArray(pathInds)
        lastActionIndex = lastActionIndex.transpose((0, 2, 1))
        lastActionIndex = lastActionIndex[:, 1:, :]
        return data, lastActionIndex
    return data


def loadTrees(path, pathInds):
    data = load3dArray(path)
    data = np.expand_dims(data, 1)
    vs = data
    if pathInds is not None:
        lastActionIndex = load3dArray(pathInds)
        assert lastActionIndex.shape == data.shape
        vs = np.concatenate([data, lastActionIndex], 1)
    return vs.reshape([vs.shape[0], vs.shape[1], -1])


def sample(data, count, inds=None):
    data = data.reshape(data.shape[0], -1).T
    inds = inds if inds is not None else np.random.choice(np.arange(data.shape[0]), count)
    return data[inds].T, inds


class TreeDatasetCompletion(torch.utils.data.Dataset):
    def __init__(self, path, overfit, missmatched=False, cache_minmax=True, group_dim=0, dtype=np.single):
        self.labels, self.map_strings = loadDataDescMap(loadDataDescStrings(path))
        print(self.labels.shape)
        self.labels = np.reshape(self.labels, (self.labels.shape[0], -1))
        self.maps = self.labels
        self.scores, self.action_ids = loadBranches(path + ".treeScores",
                                                    None if False else path + ".treeLAInds")
        self.scores = self.scores.astype(dtype=dtype)
        self.action_ids = self.action_ids.astype(dtype=dtype)
        if self.scores.shape[0] != len(self.labels):
            print(f"{self.scores.shape[0]} != {len(self.labels)}")
            self.labels = self.labels[:self.scores.shape[0]]
        sort(self.action_ids, self.scores)
        if cache_minmax and os.path.isfile(f"{path}_minmaxcache.npy"):
            self.minmaxs = np.load(f"{path}_minmaxcache.npy").astype(dtype=dtype)
        else:
            self.minmaxs = calculate_minmax(self.action_ids, self.scores)
            np.save(f"{path}_minmaxcache", self.minmaxs)
        self.minmaxs_grouped = group(self.minmaxs,
                                     self.action_ids, group_dim)
        self.action_ids_grouped = group(self.action_ids,
                                        self.action_ids, group_dim)
        if True:
            self.action_ids_grouped_onehot = as_one_hot(self.action_ids_grouped)
            self.coord_channels = self.action_ids_grouped_onehot.shape[1]
            self.data = np.concatenate([self.action_ids_grouped_onehot, self.minmaxs_grouped], 1)
        else:
            self.coord_channels = self.action_ids_grouped.shape[1]
            self.data = np.concatenate([self.action_ids_grouped, self.minmaxs_grouped], 1)
        print(self.action_ids.shape, self.action_ids_grouped.shape, self.minmaxs.shape, self.minmaxs_grouped.shape,
              self.action_ids_grouped_onehot.shape)
        if missmatched:
            np.random.shuffle(self.data)
        self.inds = np.arange(len(self.data)) if not overfit else np.arange(2)
        assert len(self.data) == len(self.labels), f"{len(self.data)} != {len(self.labels)}"

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        ind = self.inds[index]
        r = (self.data[ind],)
        return r
