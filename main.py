import torch.utils.data

from imports import *

bs = 128

use_pick_from_sides_dataset = False
missmatch = False
me = 3
reference_embeddings = False
single = True
groupingd = 2
if use_pick_from_sides_dataset:
    dataset = make_pick_from_sides_dataset(missmatch, groupingd)
else:
    path = "./data/main_data/03-26_18-24"
    #only onehot the ungrouped action dimensions
    dataset = TreeDatasetCompletion(path, False, missmatch, dtype=np.single if single else np.half, group_dim=groupingd)

print_dataset_summary(dataset, dataset.coord_channels)
len_loader = int(len(dataset) / bs)
config = {
    'experiment_name': '2_4_pointnet_classification_overfitting',
    'device': 'cuda:0',
    'batch_size': bs,
    'resume_ckpt': None,
    'learning_rate': None,
    'max_epochs': me,
    'contrastive_after': 0,
    "reference_embeddings": reference_embeddings,
    'perc_contrastive': 0.05,
    'print_every_n': 10,
    'delayed_every_n': len_loader // 2,
    "dtype": torch.float32 if single else torch.float16,
    "learning_rate_decay": None,
    "eps": 1e-8 if single else 1e-4,
    'path': None,
    'label_coords_pos': (0, dataset.coord_channels),
    "validate_every_n": len_loader
    # "visualize" : True
}

hss = [  # 2048,
    # 512,
    256,
    # 128,
    # 256+128
]
# lrs = [0.0005, 0.0001]
lrs = [0.0001]
lrds = [  # 0.95,
    0.5,
    # 0.98
]

eds = [128]
n = 5
sharpnesses = np.flip(np.linspace(5, 20, n, endpoint=True))
centering_coefs = np.flip(np.linspace(0.05, 0.98, n, endpoint=True))
np.random.shuffle(sharpnesses)
np.random.shuffle(centering_coefs)
rs = []
rshalf = []
rsquarter = []

dosl = combinations_with_replacement([0.3, 0.6], 5)
dosl = [(x[:2], tuple(y * 1.4 for y in x[2:])) for x in dosl]
np.random.shuffle(dosl)
dosl = [((0.3, 0.5), (0.5, 0.6, 0.6))]
for ed in eds:
    for dos in dosl:
        for lr in lrs:
            for lrd in lrds:
                for s in sharpnesses:
                    for c in centering_coefs:
                        r([256, ], dos, lr, lrd, rs, rshalf, rsquarter, ed, (s, c), dataset, config)
