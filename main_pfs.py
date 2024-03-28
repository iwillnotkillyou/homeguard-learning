import numpy as np
import torch.utils.data

from imports import *





overfit = False
config = {
    'experiment_name': '2_4_pointnet_classification_overfitting',
    'device': 'cuda:0',
    'batch_size': 64,
    'resume_ckpt': None,
    'learning_rate': 0.1,
    'max_epochs': me,
    'contrastive_after': 2,
    'eps': 1e-8,
    'perc_contrastive': 0.05,
    'print_every_n': 10,
    'validate_every_n': 1000,
    "learning_rate_decay": 0.9999,
    'path': None,
    'label_coords_pos': (0, coord_channels),
    # "visualize" : True
}






perc_validation = 0.20
config["validate_every_n"] = int(len(dataset) / config["batch_size"])

def badvalindsf(dataset,val_inds):
    bad_val_inds_separate = select_bad_indices_seperate(dataset.original_labels[val_inds], dataset.minmaxs[val_inds],
                                                        np.cumsum(dataset.scores, 1)[val_inds])

    bad_val_inds = np.unique(mysum(bad_val_inds_separate))
    print(bad_val_inds.shape[0] / len(val_inds))
    bad_val_inds_separate = [np.array(x) for x in bad_val_inds_separate]
    bad_val_inds = (bad_val_inds, bad_val_inds_separate)
    return bad_val_inds

def r(hs, x, y, rs, rshalf, rsquarter, missmatch):
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - perc_validation, perc_validation])
    val_inds = val_dataset.indices
    bad_val_inds = badvalindsf(dataset,val_inds)
    avg_auc = averageauc(torch.from_numpy(dataset.minmaxs[:, 0:1, 0, 0][val_inds]), bad_val_inds)
    print(f"auc_minmax_as_embedding = {avg_auc}")
    actual_channels = train_dataset[0][0].shape[0]
    assert actual_channels == total_channels, f"actual channels {actual_channels} != total_channels {total_channels}"
    config["learning_rate"] = x / 10
    config["learning_rate_decay"] = y * 0.95
    backbone = pointnet.PointNetEmbedder(total_channels, ed, hs)
    model = pointnet.PointNetAutoEncoder(backbone, coord_channels, total_channels-coord_channels, hs)
    # model = pointnet.PointNetAutoEncoder(total_channels, coord_channels, total_channels, 16, hs)
    if False:
        little = True
        l = 3 if little else 50
        m = 0 if little else 8
        config["max_epochs"] = l
        config["contrastive_after"] = l - m
    config["perc_contrastive"] = 0.1 if False else 0
    loss = train.main(model, train_dataset, val_dataset, t_multitask.train, bad_val_inds, config)
    model.eval()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size']
        , shuffle=False)
    embeds = []
    embedshalf = []
    embedquarter = []
    for batch_val in val_dataloader:
        data = batch_val[0].to(config["device"])
        with torch.no_grad():
            embeds.append(model.embedding(data))
            l = data.shape[2]
            shuffleinds = np.arange(l)
            np.random.shuffle(shuffleinds)
            embedshalf.append(model.embedding(data[:, :, shuffleinds[:l // 2]]))
            embedquarter.append(model.embedding(data[:, :, shuffleinds[:l // 4]]))
    embeds = torch.concatenate(embeds, 0)
    embedshalf = torch.concatenate(embedshalf, 0)
    embedquarter = torch.concatenate(embedquarter, 0)
    maps = dataset.original_labels[val_inds]
    print("--full--")
    print_info(embeds, bad_val_inds, loss, (x, y, hs), rs, maps)
    print("--half--")
    print_info(embedshalf, bad_val_inds, loss, (x, y, hs), rshalf, maps)
    print("--quarter--")
    print_info(embedquarter, bad_val_inds, loss, (x, y, hs), rsquarter, maps)
    return backbone


def rDino(hs, x, y, s, c, rs, backbone, missmatch):
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - perc_validation, perc_validation])
    val_inds = val_dataset.indices
    bad_val_inds = badvalindsf(dataset,val_inds)
    actual_channels = train_dataset[0][0].shape[0]
    assert actual_channels == total_channels, f"actual channels {actual_channels} != total_channels {total_channels}"
    config["learning_rate"] = x
    config["learning_rate_decay"] = y
    config["embedding_dim"] = ed
    config["sharpness"] = s
    config["center_momentum"] = c
    # backbone = pointnet.PointNetEmbedder(total_channels, hs//4, hs)
    model = pointnet.PointNetDino(backbone, (ed * 2,), ed)
    loss = train.main(model, train_dataset, val_dataset, t_dino.train, bad_val_inds, config)
    model.eval()
    embeds = []
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size']
        , shuffle=False)
    for batch_val in val_dataloader:
        data = batch_val[0].to(config["device"])
        with torch.no_grad():
            embeds.append(model(data))
    embeds = torch.concatenate(embeds, 0).unsqueeze(0)
    print_info(embeds, bad_val_inds, loss, (x, y, hs, s, c), rs)


##ADD LOSS CONTRIBUTION SCHEDULING AT FIRST ONLY COMPLETION, ADD CLASSIFICATION TO MULTITASK
##TRY DIFFERENT TRUNCATION METHODS - IE SOFTMAX, SUM, TRY ATTENTION IN LATER LAYERS, TRY CHANGING MAX AND LENGTH
# TRY ADDING SEPERATION TO COMPLETION

## REMEBMER THAT THE MAIN TASK IS  CLUSTERING USING EMBEDDINGS NOT THE PRETEXT TASKS
## USE L1 or sqrt loss to make hard examples more important in the loss use L2 for inner average amd sqrt for outer average
# derivative might not make sense

# Do similarity based pooling on deeper layers

# At some stage add aversarial training loss
## at a DINO like knowledge distilation pass the last 8-4 layers to the student together that is there will be some layer where all node's
# children will make up the training data so num of views == number of nodes at layer
# pass the whole tree to the teacher - maybe do 1/2 on teacher and 1/4 on student and do completion on both as pretext
# use classification head output or latent space for the taught layer
## after fully implementing the learning paradigm make sure to check for correctness (add a fuckton of asserts and tests)

## learn a tree representation using a message passing network - use global feature concatted with random vector or tree coordinates
## and initialization network to initialize - then encode this representation in a vector - maybe use multiple decreasing sizes
## read up more on message passing networks
hss = [  # 2048,
    # 1024,
    # 516,
    # 64,
    128,
    # 256
]
lrs = [0.0005]
lrds = [  # 0.95,
    0.97,
    0.98
]
sharpnesses = np.linspace(1, 20, 5, endpoint=True)
centering_coefs = np.linspace(0.5, 0.98, 5, endpoint=True)
rs = []
rshalf = []
rsquarter = []
for hs in hss:
    for x in lrs:
        for y in lrds:
            for s in sharpnesses:
                for c in centering_coefs:
                    if True:
                        r(hs, x, y, rs, rshalf, rsquarter, missmatch)
                    else:
                        backbone = r(hs, x, y, [], [], [], missmatch)
                        rDino(hs, x, y, s, c, rs, backbone, missmatch)
