from imports import *


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, dataF, labelsF, f=lambda x, y: (x, y), missmatch=False):
        self.original_data = load3dArray(dataF)
        self.f = f
        self.original_labels = load3dArray(labelsF)[..., 0].astype(np.dtype("int64"))
        self.data, self.labels = f(self.original_data, self.original_labels)
        self.ordering = np.arange(self.data.shape[0])
        if missmatch:
            np.random.shuffle(self.ordering)
            self.data = self.data[self.ordering]
        self.unshuf_ordering = np.zeros_like(self.ordering)
        self.unshuf_ordering[self.ordering] = np.arange(self.data.shape[0])
        assert len(self.data) == len(self.labels)
        self.num_classes = int(np.max(self.labels) + 1)
        self.vector_length = self.labels.shape[1]
        assert not np.isnan(self.num_classes), f"labels : {np.unique(self.labels)},data : {np.unique(self.data)}"
        assert not np.isnan(np.max(self.data)), f"labels : {np.unique(self.labels)},data : {np.unique(self.data)}"
        self.inds = np.arange(len(self.data))

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[self.inds[index]]),
                torch.from_numpy(self.labels[self.inds[index]]))


class FileDatasetMinmax(torch.utils.data.Dataset):
    def __init__(self, dataF, labelsF, f=lambda x, y, z: (x, z), coord_channels=1, missmatch=False):
        self.original_data = load3dArray(dataF)
        self.f = f
        self.coord_channels = coord_channels
        self.original_labels = load3dArray(labelsF)[..., 0].astype(np.dtype("int64"))
        leng = self.original_labels.shape[1]
        self.actions = self.original_data[:, :leng]
        self.scores = self.original_data[:, leng:2 * leng]
        self.minmaxs = self.original_data[:, 2 * leng:]
        self.minmaxs = self.minmaxs.reshape([self.minmaxs.shape[0], leng, self.minmaxs.shape[2], -1])
        self.data, self.labels = f(self.actions, self.scores, self.minmaxs)
        self.ordering = np.arange(self.data.shape[0])
        self.map_strings = np.array([str(x) for x in np.squeeze(self.original_labels)])
        self.maps = np.squeeze(self.original_labels)
        if missmatch:
            np.random.shuffle(self.ordering)
            self.data = self.data[self.ordering]
        self.unshuf_ordering = np.zeros_like(self.ordering)
        self.unshuf_ordering[self.ordering] = np.arange(self.data.shape[0])
        assert len(self.data) == len(self.labels)
        self.num_classes = int(np.max(self.labels) + 1)
        self.vector_length = self.labels.shape[1]
        assert not np.isnan(self.num_classes), f"labels : {np.unique(self.labels)},data : {np.unique(self.data)}"
        assert not np.isnan(np.max(self.data)), f"labels : {np.unique(self.labels)},data : {np.unique(self.data)}"
        self.inds = np.arange(len(self.data))

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[self.inds[index]]),
                torch.from_numpy(self.labels[self.inds[index]]))
